"""
Model conversion

Usage:
    conversions.py <model_state_path>  <inputs_path> <outputs_path> [--fp16_mode]

"""

import pickle

import torch
import numpy as np
import tensorrt as trt
from docopt import docopt
from trt_helper import TRTInferenceModule


def conversions(model_state_path, inputs_path, outputs_path, fp16_mode):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    plg_registry = trt.get_plugin_registry()

    # qkv plugin
    qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "1", "")
    pf_type = trt.PluginField("type_id", np.array([fp16_mode], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([768], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([12], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    # gelu plugin
    gelu_plg_creator = plg_registry.get_plugin_creator("CustomGeluPluginDynamic", "1", "")
    pf_type = trt.PluginField("type_id", np.array([fp16_mode], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_type, ])
    gelu_plug = gelu_plg_creator.create_plugin("gelu_plugin", pfc)

    # build network
    builder = trt.Builder(TRT_LOGGER)
    builder.max_workspace_size = 2 << 31
    builder.fp16_mode = fp16_mode
    network = builder.create_network((1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))

    clf_token = network.add_input(name='clf_token', dtype=trt.float32, shape=[1, 1, 768])
    patch_embeddings_TSC11 = network.add_input(name='patch_embeddings_TSC11', dtype=trt.float32,
                                               shape=[8, 196, 768, 1, 1])

    # auxilary constants
    ones_constant_8_1_768 = np.ones([8, 1, 768]).astype(np.float32)
    ones_constant_8_1_768 = trt.Weights(ones_constant_8_1_768)
    ones_constant_8_1_768 = network.add_constant(trt.Dims((8, 1, 768)), ones_constant_8_1_768).get_output(0)

    # model weights
    model_state_dict = torch.load(model_state_path, map_location='cpu')['model_state']

    def layer_norm(network, inputs, weights, bias, axes=4):
        # compute mean
        mean = network.add_reduce(inputs, trt.ReduceOperation.AVG, axes=axes, keep_dims=True).get_output(0)

        # compute diff
        diff = network.add_elementwise(inputs, mean, trt.ElementWiseOperation.SUB).get_output(0)

        # compute std
        x = network.add_elementwise(diff, diff, trt.ElementWiseOperation.PROD).get_output(0)
        x = network.add_reduce(x, trt.ReduceOperation.AVG, axes=axes, keep_dims=True).get_output(0)
        std = network.add_unary(x, trt.UnaryOperation.SQRT).get_output(0)

        # compute normalized inputs
        normalized_inputs = network.add_elementwise(diff, std, trt.ElementWiseOperation.DIV).get_output(0)

        # gamma
        normalized_inputs = network.add_elementwise(normalized_inputs, weights,
                                                    trt.ElementWiseOperation.PROD).get_output(0)

        #     # beta
        normalized_inputs = network.add_elementwise(normalized_inputs, bias, trt.ElementWiseOperation.SUM).get_output(0)

        return normalized_inputs

    def build_block(idx, patch_embeddings_TSC11, clf_token):
        # step 1 temporal attention

        # layer normalization
        weights = model_state_dict['model.blocks.{0}.temporal_norm1.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        weights = network.add_constant(trt.Dims((1, 1, 768, 1, 1)), weights).get_output(0)
        bias = model_state_dict['model.blocks.{0}.temporal_norm1.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        bias = network.add_constant(trt.Dims((1, 1, 768, 1, 1)), bias).get_output(0)
        normalized_patch_embeddings_TSC11 = layer_norm(network, patch_embeddings_TSC11, weights, bias, axes=4)

        # compute qkv
        weights = model_state_dict['model.blocks.{0}.temporal_attn.qkv.weight'.format(idx)].reshape(3, 12, 64, 768)
        weights = weights.permute(1, 0, 2, 3).reshape(2304, 768)
        weights = weights.numpy().astype(np.float32)
        weights = trt.Weights(weights)

        bias = model_state_dict['model.blocks.{0}.temporal_attn.qkv.bias'.format(idx)].reshape(3, 12, 64)
        bias = bias.permute(1, 0, 2).reshape(2304, )
        bias = bias.numpy().astype(np.float32)
        bias = trt.Weights(bias)

        qkv = network.add_fully_connected(normalized_patch_embeddings_TSC11, 2304, weights, bias).get_output(0)

        # get self attention output
        qkv2ctx = network.add_plugin_v2([qkv], qkv2ctx_plug)
        self_attention_output = qkv2ctx.get_output(0)

        # FC on the output
        weights = model_state_dict['model.blocks.{0}.temporal_attn.proj.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.temporal_attn.proj.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        self_attention_fc_output = network.add_fully_connected(self_attention_output, 768, weights, bias).get_output(0)

        weights = model_state_dict['model.blocks.{0}.temporal_fc.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.temporal_fc.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        self_attention_fc_output = network.add_fully_connected(self_attention_fc_output, 768, weights, bias).get_output(
            0)

        # add the self attention output with original input patch embeddings
        patch_embeddings_TSC11_T = network.add_elementwise(
            patch_embeddings_TSC11,
            self_attention_fc_output,
            trt.ElementWiseOperation.SUM).get_output(0)

        # repeat clf token from 1x1xC to Tx1xC
        clf_token_T1C = network.add_elementwise(clf_token, ones_constant_8_1_768,
                                                trt.ElementWiseOperation.PROD).get_output(
            0)

        # concatenate clf tokens with patch embeddings after temporal attention,
        shuffle_layer = network.add_shuffle(patch_embeddings_TSC11_T)
        shuffle_layer.reshape_dims = (8, 196, 768)
        patch_embeddings_TSC_T = shuffle_layer.get_output(0)

        concat_layer = network.add_concatenation([clf_token_T1C, patch_embeddings_TSC_T])
        concat_layer.axis = 1
        patch_embeddings_TSC_T_including_clf = concat_layer.get_output(0)

        shuffle_layer = network.add_shuffle(patch_embeddings_TSC_T_including_clf)
        shuffle_layer.reshape_dims = (8, 197, 768, 1, 1)
        patch_embeddings_TSC11_T_including_clf = shuffle_layer.get_output(0)

        # Do spatial attention

        # layer normalization
        weights = model_state_dict['model.blocks.{0}.norm1.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        weights = network.add_constant(trt.Dims((1, 1, 768, 1, 1)), weights).get_output(0)
        bias = model_state_dict['model.blocks.{0}.norm1.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        bias = network.add_constant(trt.Dims((1, 1, 768, 1, 1)), bias).get_output(0)
        patch_embeddings_TSC11_T_including_clf_normalized = layer_norm(network, patch_embeddings_TSC11_T_including_clf,
                                                                       weights, bias, axes=4)

        # TSC11 -> STC11
        shuffle_layer = network.add_shuffle(patch_embeddings_TSC11_T_including_clf_normalized)
        shuffle_layer.first_transpose = trt.Permutation([1, 0, 2, 3, 4])
        patch_embeddings_STC11_T_including_clf_normalized = shuffle_layer.get_output(0)

        # qkv
        weights = model_state_dict['model.blocks.{0}.attn.qkv.weight'.format(idx)].reshape(3, 12, 64, 768)
        weights = weights.permute(1, 0, 2, 3).reshape(2304, 768)
        weights = weights.numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.attn.qkv.bias'.format(idx)].reshape(3, 12, 64)
        bias = bias.permute(1, 0, 2).reshape(2304, )
        bias = bias.numpy().astype(np.float32)
        bias = trt.Weights(bias)
        qkv = network.add_fully_connected(patch_embeddings_STC11_T_including_clf_normalized, 2304, weights,
                                          bias).get_output(0)

        # get self attention output
        qkv2ctx = network.add_plugin_v2([qkv], qkv2ctx_plug)
        self_attention_output = qkv2ctx.get_output(0)
        # fc to self attention
        weights = model_state_dict['model.blocks.{0}.attn.proj.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.attn.proj.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        self_attention_fc_output = network.add_fully_connected(self_attention_output, 768, weights, bias).get_output(0)

        # split the cls tokens and rest from the output
        clf_tokens_1TC11 = network.add_slice(self_attention_fc_output, start=(0, 0, 0, 0, 0), shape=(1, 8, 768, 1, 1),
                                             stride=(1, 1, 1, 1, 1)).get_output(0)
        self_attention_output_STC11 = network.add_slice(self_attention_fc_output, start=(1, 0, 0, 0, 0),
                                                        shape=(196, 8, 768, 1, 1), stride=(1, 1, 1, 1, 1)).get_output(0)

        # STC11 -> TSC11 ->NC11
        shuffle_layer = network.add_shuffle(self_attention_output_STC11)
        shuffle_layer.first_transpose = trt.Permutation([1, 0, 2, 3, 4])
        shuffle_layer.reshape_dims = (1568, 768, 1, 1)
        self_attention_output_NC11 = shuffle_layer.get_output(0)

        # compute mean  cls tokens
        clf_token_mean = network.add_reduce(clf_tokens_1TC11, trt.ReduceOperation.AVG, axes=2,
                                            keep_dims=False).get_output(
            0)

        # add the mean clf token back to the self attention output -> (1+TS x C x 1 x 1)
        concat_layer = network.add_concatenation([clf_token_mean, self_attention_output_NC11])
        concat_layer.axis = 0
        self_attention_output_NC11_including_clf = concat_layer.get_output(0)

        # original clf token 11C -> 1C11
        shuffle_layer = network.add_shuffle(clf_token)
        shuffle_layer.reshape_dims = (1, 768, 1, 1)
        clf_token_1C11 = shuffle_layer.get_output(0)

        # patch embedding after temporal attetion TSC11->NC11
        shuffle_layer = network.add_shuffle(patch_embeddings_TSC11_T)
        shuffle_layer.reshape_dims = (1568, 768, 1, 1)
        patch_embeddings_NC11_T = shuffle_layer.get_output(0)

        # concatenate original clf token to above
        concat_layer = network.add_concatenation([clf_token_1C11, patch_embeddings_NC11_T])
        concat_layer.axis = 0
        patch_embeddings_NC11_T_including_clf = concat_layer.get_output(0)

        # add to the self attention residual
        patch_embeddings_NC11_S_including_clf = network.add_elementwise(
            self_attention_output_NC11_including_clf,
            patch_embeddings_NC11_T_including_clf,
            trt.ElementWiseOperation.SUM).get_output(0)

        # layer normalization
        weights = model_state_dict['model.blocks.{0}.norm2.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        weights = network.add_constant(trt.Dims((1, 768, 1, 1)), weights).get_output(0)
        bias = model_state_dict['model.blocks.{0}.norm2.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        bias = network.add_constant(trt.Dims((1, 768, 1, 1)), bias).get_output(0)
        patch_embeddings_NC11_S_including_clf_normalized = layer_norm(network, patch_embeddings_NC11_S_including_clf,
                                                                      weights, bias, axes=2)

        # mlp
        weights = model_state_dict['model.blocks.{0}.mlp.fc1.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.mlp.fc1.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        fc1_output = network.add_fully_connected(patch_embeddings_NC11_S_including_clf_normalized, 3072, weights,
                                                 bias).get_output(0)

        shuffle_layer = network.add_shuffle(fc1_output)
        shuffle_layer.reshape_dims = (1, 1569, 3072)
        fc1_output = shuffle_layer.get_output(0)

        gelu = network.add_plugin_v2([fc1_output], gelu_plug).get_output(0)

        shuffle_layer = network.add_shuffle(gelu)
        shuffle_layer.reshape_dims = (1569, 3072, 1, 1)
        gelu = shuffle_layer.get_output(0)

        weights = model_state_dict['model.blocks.{0}.mlp.fc2.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.mlp.fc2.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        fc2_output = network.add_fully_connected(gelu, 768, weights, bias).get_output(0)
        # fc2_output = network.add_fully_connected(fc1_output, 768, weights, bias).get_output(0)

        patch_embeddings_NC11_mlp_including_clf = network.add_elementwise(
            fc2_output,
            patch_embeddings_NC11_S_including_clf,
            trt.ElementWiseOperation.SUM).get_output(0)

        patch_embeddings_NC11_mlp = network.add_slice(
            patch_embeddings_NC11_mlp_including_clf,
            start=(1, 0, 0, 0),
            shape=(1568, 768, 1, 1),
            stride=(1, 1, 1, 1)).get_output(0)
        shuffle_layer = network.add_shuffle(patch_embeddings_NC11_mlp)
        shuffle_layer.reshape_dims = (8, 196, 768, 1, 1)
        patch_embeddings = shuffle_layer.get_output(0)

        clf_token_1C11 = network.add_slice(
            patch_embeddings_NC11_mlp_including_clf,
            start=(0, 0, 0, 0),
            shape=(1, 768, 1, 1),
            stride=(1, 1, 1, 1)).get_output(0)
        shuffle_layer = network.add_shuffle(clf_token_1C11)
        shuffle_layer.reshape_dims = (1, 1, 768)
        clf_token = shuffle_layer.get_output(0)

        return patch_embeddings_TSC11_T, patch_embeddings_NC11_S_including_clf, fc1_output, gelu, fc2_output, patch_embeddings, clf_token

    for idx in range(12):
        patch_embeddings_TSC11_T, patch_embeddings_NC11_S_including_clf, fc1_output, gelu, fc2_output, patch_embeddings_TSC11, clf_token = build_block(
            idx, patch_embeddings_TSC11, clf_token)
        network.mark_output(patch_embeddings_TSC11_T)
        network.mark_output(patch_embeddings_NC11_S_including_clf)
        network.mark_output(fc1_output)
        network.mark_output(gelu)
        network.mark_output(fc2_output)
        network.mark_output(patch_embeddings_TSC11)

    engine = builder.build_cuda_engine(network)

    inference = TRTInferenceModule(engine)
    inputs = pickle.load(open(inputs_path, 'rb'))
    outputs = inference.do_inference(inputs)

    pickle.dump(outputs, open(outputs_path, 'wb'))


if __name__ == '__main__':
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)

    model_state_path = arguments['<model_state_path>']
    inputs_path = arguments['<inputs_path>']
    outputs_path = arguments['<outputs_path>']
    fp16_mode = arguments['--fp16_mode']

    conversions(model_state_path, inputs_path, outputs_path, fp16_mode)
