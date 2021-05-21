import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, raw_shape):
        self.host = host_mem
        self.device = device_mem
        self.raw_shape = raw_shape
        self.flatten_size = np.prod(self.raw_shape[1:])

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def to_original_shape(self):
        return self.host.reshape(self.raw_shape)


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        raw_shape = [engine.max_batch_size] + list(engine.get_binding_shape(binding))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, raw_shape))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, raw_shape))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.to_original_shape()[:batch_size] for out in outputs]


class TRTInferenceModule(object):

    def __init__(self, engine):
        # load engine, allocate context
        self.engine = engine
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.max_batch_size = self.engine.max_batch_size

    def load_numpy_inputs(self, inputs_list):

        batch_size = inputs_list[0].shape[0]

        for idx, inputs in enumerate(inputs_list):
            inputs_flatten = inputs.flatten()
            input_host = self.inputs[idx].host
            if batch_size < self.max_batch_size:
                input_host = input_host[:batch_size * self.inputs[idx].flatten_size]
            else:
                input_host = self.inputs[idx].host
            np.copyto(input_host, inputs_flatten)

    def do_inference(self, inputs_list):

        # load input to gpu
        self.load_numpy_inputs(inputs_list)

        # inference
        batch_size = inputs_list[0].shape[0]
        outputs = do_inference(self.context, self.bindings, self.inputs, self.outputs, self.stream,
                               batch_size=batch_size)

        return outputs
