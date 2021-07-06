"""
Model conversion

Usage:
    conversions.py <model_state_path> <test_data_path> <outputs_engine_path> <test_result_path> [--fp16_mode] [--expose_ln_entries]

Options:
    --fp16_mode             run in FP16
    --expose_ln_entries     expose Layer norm entries in the 12 T-S blocks as outputs

"""

import pickle

import torch
import numpy as np
import tensorrt as trt
from docopt import docopt
from trt_helper import TRTInferenceModule


def conversions(model_state_path, test_data_path, outputs_engine_path, test_result_path, fp16_mode,
                expose_layer_norm_entries):
    """

    :param model_state_path: pytorch model path
    :param test_data_path: test data path
    :param outputs_engine_path: outputs engine path
    :param test_result_path: outputs data path
    :param fp16_mode: build in FP16
    :param expose_layer_norm_entries: whether expose layer norm entries in outputs
    :return: None
    """
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

    # input video shape TxCxHxW
    input_tensor = network.add_input(name='inputs', dtype=trt.float32, shape=[8, 3, 224, 224])

    # auxilary constant used later in network building
    ones_constant_8_1_768 = np.ones([8, 1, 768]).astype(np.float32)
    ones_constant_8_1_768 = trt.Weights(ones_constant_8_1_768)
    ones_constant_8_1_768 = network.add_constant(trt.Dims((8, 1, 768)), ones_constant_8_1_768).get_output(0)

    # load model weights
    model_state_dict = torch.load(model_state_path, map_location='cpu')['model_state']
    # class token
    clf_token = model_state_dict['model.cls_token'].numpy().astype(np.float32)

    # time position embedding
    time_embed = model_state_dict['model.time_embed'].numpy().astype(np.float32)
    # spatial position embedding
    pos_embed = model_state_dict['model.pos_embed'].numpy().astype(np.float32)

    # add the first spatial pos emb to class token
    clf_token = clf_token + pos_embed[:, :1, :]
    clf_token = trt.Weights(clf_token[0])
    # reshape to 1x1xC
    clf_token = network.add_constant(trt.Dims((1, 1, 768)), clf_token).get_output(0)

    # create pos+time pos embedding matrix for input video, output shape TS x C
    pos_embed = pos_embed[:, 1:, :]
    time_embed = time_embed.reshape(8, 1, 768)
    pos_embed = pos_embed + time_embed
    pos_embed = pos_embed.reshape(1568, 768)
    pos_embed_trt = trt.Weights(pos_embed)
    pos_embed_trt = network.add_constant(trt.Dims((1568, 768)), pos_embed_trt)
    pos_embed_trt = pos_embed_trt.get_output(0)

    # generate spatial patches features
    patch_proj_weight = model_state_dict['model.patch_embed.proj.weight'].numpy().astype(np.float32)
    patch_proj_weight = trt.Weights(patch_proj_weight)

    patch_proj_bias = model_state_dict['model.patch_embed.proj.bias'].numpy().astype(np.float32)
    patch_proj_bias = trt.Weights(patch_proj_bias)

    patch_proj_layer = network.add_convolution(input_tensor, num_output_maps=768, kernel_shape=(16, 16),
                                               kernel=patch_proj_weight, bias=patch_proj_bias)
    patch_proj_layer.stride = (16, 16)
    patch_embeddings = patch_proj_layer.get_output(0)
    shuffle_layer = network.add_shuffle(patch_embeddings)
    shuffle_layer.first_transpose = trt.Permutation([0, 2, 3, 1])
    shuffle_layer.reshape_dims = (1568, 768)
    patch_embeddings = shuffle_layer.get_output(0)

    addition = network.add_elementwise(patch_embeddings, pos_embed_trt, trt.ElementWiseOperation(0))
    patch_embeddings = addition.get_output(0)

    # VALUES ARE IN HALF
    # reshape patch embedding to T x S x C x 1 x 1
    shuffle_layer = network.add_shuffle(patch_embeddings)
    shuffle_layer.reshape_dims = (8, 196, 768, 1, 1)
    patch_embeddings_TSC11 = shuffle_layer.get_output(0)

    # helper fn: layer norm
    def layer_norm(network, inputs, weights, bias, block_num, ln_num, axes=4):
        # compute mean
        mean = network.add_reduce(inputs, trt.ReduceOperation.AVG, axes=axes, keep_dims=True)
        mean.name = '{0}_{1}_mean_inputs'.format(block_num, ln_num)
        mean = mean.get_output(0)

        # compute diff
        diff = network.add_elementwise(inputs, mean, trt.ElementWiseOperation.SUB)
        diff.name = '{0}_{1}_input_sub_mean'.format(block_num, ln_num)
        diff = diff.get_output(0)

        # compute std
        POW = network.add_constant((1,) * len(diff.shape), trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32)))
        POW.name = '{0}_{1}_power_const'.format(block_num, ln_num)

        x = network.add_elementwise(diff, POW.get_output(0), trt.ElementWiseOperation.POW)
        x.name = '{0}_{1}_squared_diff'.format(block_num, ln_num)
        x = x.get_output(0)

        x = network.add_reduce(x, trt.ReduceOperation.AVG, axes=axes, keep_dims=True)
        x.name = '{0}_{1}_mean_squared_diff'.format(block_num, ln_num)
        x = x.get_output(0)
        eps = np.array([1e-6])
        eps = eps.astype(np.float32)
        x = network.add_scale(x, mode=trt.ScaleMode.ELEMENTWISE, shift=trt.Weights(eps))
        x.name = '{0}_{1}_add_eps'.format(block_num, ln_num)
        x = x.get_output(0)

        std = network.add_unary(x, trt.UnaryOperation.SQRT)
        std.name = '{0}_{1}_std'.format(block_num, ln_num)
        std = std.get_output(0)

        # compute normalized inputs
        normalized_inputs = network.add_elementwise(diff, std, trt.ElementWiseOperation.DIV)
        normalized_inputs.name = '{0}_{1}_diff_div_std'.format(block_num, ln_num)
        normalized_inputs = normalized_inputs.get_output(0)

        # weights bias
        normalized_inputs = network.add_scale(normalized_inputs, mode=trt.ScaleMode.CHANNEL, scale=trt.Weights(weights),
                                              shift=trt.Weights(bias))
        normalized_inputs.name = '{0}_{1}_rescale'.format(block_num, ln_num)
        normalized_inputs = normalized_inputs.get_output(0)
        return mean, diff, std, normalized_inputs

    def build_block(idx, patch_embeddings_TSC11, clf_token):
        # step 1 temporal attention

        # layer normalization
        weights = model_state_dict['model.blocks.{0}.temporal_norm1.weight'.format(idx)].numpy().astype(np.float32)
        bias = model_state_dict['model.blocks.{0}.temporal_norm1.bias'.format(idx)].numpy().astype(np.float32)
        layernorm_mean, layernorm_diff, layernorm_std, normalized_patch_embeddings_TSC11 = layer_norm(network,
                                                                                                      patch_embeddings_TSC11,
                                                                                                      weights, bias,
                                                                                                      idx, 0,
                                                                                                      axes=4)

        # 1.2 self attention
        # 1.2.1 compute qkv
        # the original qkv weights are in order of 3 x NUM_HEADS x HEAD_DIM x Channels
        # need to convert to NUM_HEADS x 3 x HEAD_DIM x Channels
        weights = model_state_dict['model.blocks.{0}.temporal_attn.qkv.weight'.format(idx)].reshape(3, 12, 64, 768)
        weights = weights.permute(1, 0, 2, 3).reshape(2304, 768)
        weights = weights.numpy().astype(np.float32)
        weights = trt.Weights(weights)

        # same for the bias
        bias = model_state_dict['model.blocks.{0}.temporal_attn.qkv.bias'.format(idx)].reshape(3, 12, 64)
        bias = bias.permute(1, 0, 2).reshape(2304, )
        bias = bias.numpy().astype(np.float32)
        bias = trt.Weights(bias)

        # generate qkv per patch => Tx S x NUM_HEADS-3-HEAD_DIM-CHANNELS x 1 x 1
        temporal_attention_qkv = network.add_fully_connected(normalized_patch_embeddings_TSC11, 2304, weights,
                                                             bias).get_output(0)

        # get self attention output
        qkv2ctx = network.add_plugin_v2([temporal_attention_qkv], qkv2ctx_plug)
        temporal_attention_sa = qkv2ctx.get_output(0)

        # 1.2.3 run projection
        weights = model_state_dict['model.blocks.{0}.temporal_attn.proj.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.temporal_attn.proj.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        temporal_attention_proj = network.add_fully_connected(temporal_attention_sa, 768, weights, bias).get_output(0)

        # 1.3 run temporal FC to produce the residual
        weights = model_state_dict['model.blocks.{0}.temporal_fc.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.temporal_fc.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        temporal_attention_residual = network.add_fully_connected(temporal_attention_proj, 768, weights,
                                                                  bias).get_output(0)

        # 1.4 add the residual to original embedding to form patch embeddding after temporal transform
        patch_embeddings_TSC11_T = network.add_elementwise(
            patch_embeddings_TSC11,
            temporal_attention_residual,
            trt.ElementWiseOperation.SUM).get_output(0)

        # 2 spatial attention
        # 2.1 repeat clf token from 1x1xC to Tx1xC
        clf_token_T1C = network.add_elementwise(clf_token, ones_constant_8_1_768,
                                                trt.ElementWiseOperation.PROD).get_output(0)

        # 2.2 concatenate clf tokens with patch embeddings after temporal attention,
        shuffle_layer = network.add_shuffle(patch_embeddings_TSC11_T)
        shuffle_layer.reshape_dims = (8, 196, 768)
        patch_embeddings_TSC_T = shuffle_layer.get_output(0)

        concat_layer = network.add_concatenation([clf_token_T1C, patch_embeddings_TSC_T])
        concat_layer.axis = 1
        patch_embeddings_TSC_T_including_clf = concat_layer.get_output(0)

        shuffle_layer = network.add_shuffle(patch_embeddings_TSC_T_including_clf)
        shuffle_layer.reshape_dims = (8, 197, 768, 1, 1)
        patch_embeddings_TSC11_T_including_clf = shuffle_layer.get_output(0)

        # 2.3 run layer normalization
        weights = model_state_dict['model.blocks.{0}.norm1.weight'.format(idx)].numpy().astype(np.float32)
        bias = model_state_dict['model.blocks.{0}.norm1.bias'.format(idx)].numpy().astype(np.float32)
        layernorm_mean_2, layernorm_diff_2, layernorm_std_2, patch_embeddings_TSC11_T_including_clf_normalized = layer_norm(
            network, patch_embeddings_TSC11_T_including_clf,
            weights, bias, idx, 1, axes=4)

        # 2.4 do spatial self-attention

        # 2.4.1 reshufle the axis: TSC11 -> STC11
        shuffle_layer = network.add_shuffle(patch_embeddings_TSC11_T_including_clf_normalized)
        shuffle_layer.first_transpose = trt.Permutation([1, 0, 2, 3, 4])
        patch_embeddings_STC11_T_including_clf_normalized = shuffle_layer.get_output(0)

        # 2.4.2 qkv
        weights = model_state_dict['model.blocks.{0}.attn.qkv.weight'.format(idx)].reshape(3, 12, 64, 768)
        weights = weights.permute(1, 0, 2, 3).reshape(2304, 768)
        weights = weights.numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.attn.qkv.bias'.format(idx)].reshape(3, 12, 64)
        bias = bias.permute(1, 0, 2).reshape(2304, )
        bias = bias.numpy().astype(np.float32)
        bias = trt.Weights(bias)
        spatial_attention_qkv = network.add_fully_connected(patch_embeddings_STC11_T_including_clf_normalized, 2304,
                                                            weights,
                                                            bias).get_output(0)

        # 2.4.3 run self attention
        qkv2ctx = network.add_plugin_v2([spatial_attention_qkv], qkv2ctx_plug)
        spatial_attention_sa = qkv2ctx.get_output(0)

        # 2.4.4 run projection
        weights = model_state_dict['model.blocks.{0}.attn.proj.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.attn.proj.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        spatial_attention_proj = network.add_fully_connected(spatial_attention_sa, 768, weights, bias).get_output(0)

        # 2.6, for projection, only use the average token projecction from the T cls token projections.

        # 2.6.1 split the cls tokens and rest from the projection
        clf_tokens_1TC11 = network.add_slice(spatial_attention_proj, start=(0, 0, 0, 0, 0), shape=(1, 8, 768, 1, 1),
                                             stride=(1, 1, 1, 1, 1)).get_output(0)
        self_attention_output_STC11 = network.add_slice(spatial_attention_proj, start=(1, 0, 0, 0, 0),
                                                        shape=(196, 8, 768, 1, 1), stride=(1, 1, 1, 1, 1)).get_output(0)

        # 2.6.2 reshuffle rest of patches projection STC11 -> TSC11 ->NC11
        shuffle_layer = network.add_shuffle(self_attention_output_STC11)
        shuffle_layer.first_transpose = trt.Permutation([1, 0, 2, 3, 4])
        shuffle_layer.reshape_dims = (1568, 768, 1, 1)
        self_attention_output_NC11 = shuffle_layer.get_output(0)

        # 2.6.3 Average the T cls tokens
        clf_token_mean = network.add_reduce(clf_tokens_1TC11, trt.ReduceOperation.AVG, axes=2,
                                            keep_dims=False).get_output(0)

        # 2.6.4  add the mean clf token back to patches projection to form spatial attention residuals -> (1+TS x C x 1 x 1)
        concat_layer = network.add_concatenation([clf_token_mean, self_attention_output_NC11])
        concat_layer.axis = 0
        spatial_attention_residual = concat_layer.get_output(0)

        # 2.7 add back the residual

        # 2.7.1 concatenate orignal clf token to patch embeddings after T-attention in form of ((1+TS) X C X 1 X 1)

        # 2.7.1.1 original clf token 11C -> 1C11
        shuffle_layer = network.add_shuffle(clf_token)
        shuffle_layer.reshape_dims = (1, 768, 1, 1)
        clf_token_1C11 = shuffle_layer.get_output(0)

        # 2.7.1.2 patch embedding after temporal attetion TSC11->NC11
        shuffle_layer = network.add_shuffle(patch_embeddings_TSC11_T)
        shuffle_layer.reshape_dims = (1568, 768, 1, 1)
        patch_embeddings_NC11_T = shuffle_layer.get_output(0)

        # 2.7.1.3 concatenate original clf token to above
        concat_layer = network.add_concatenation([clf_token_1C11, patch_embeddings_NC11_T])
        concat_layer.axis = 0
        patch_embeddings_NC11_T_including_clf = concat_layer.get_output(0)

        # 2.7.2 add to the self attention residual
        patch_embeddings_NC11_S_including_clf = network.add_elementwise(
            spatial_attention_residual,
            patch_embeddings_NC11_T_including_clf,
            trt.ElementWiseOperation.SUM).get_output(0)

        # 2.8 layer normalization
        weights = model_state_dict['model.blocks.{0}.norm2.weight'.format(idx)].numpy().astype(np.float32)
        bias = model_state_dict['model.blocks.{0}.norm2.bias'.format(idx)].numpy().astype(np.float32)
        layernorm_mean_3, layernorm_diff_3, layernorm_std_3, patch_embeddings_NC11_S_including_clf_normalized = layer_norm(
            network, patch_embeddings_NC11_S_including_clf,
            weights, bias, idx, 2, axes=2)

        # 2.9 mlp block

        # 2.9.1 fc1
        weights = model_state_dict['model.blocks.{0}.mlp.fc1.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.mlp.fc1.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        fc1_output = network.add_fully_connected(patch_embeddings_NC11_S_including_clf_normalized, 3072, weights,
                                                 bias).get_output(0)

        # 2.9.2 non linearity - gelu
        gelu = network.add_plugin_v2([fc1_output], gelu_plug).get_output(0)

        # 2.9.3 fc2
        weights = model_state_dict['model.blocks.{0}.mlp.fc2.weight'.format(idx)].numpy().astype(np.float32)
        weights = trt.Weights(weights)
        bias = model_state_dict['model.blocks.{0}.mlp.fc2.bias'.format(idx)].numpy().astype(np.float32)
        bias = trt.Weights(bias)
        fc2_output = network.add_fully_connected(gelu, 768, weights, bias).get_output(0)

        # 2.9.4 add fc2 back to inputs
        patch_embeddings_NC11_mlp_including_clf = network.add_elementwise(
            fc2_output,
            patch_embeddings_NC11_S_including_clf,
            trt.ElementWiseOperation.SUM).get_output(0)

        # 2.9.5 split embeddings to spatial temporal TxSXCX1X1 and cls token 1X1XC
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

        # SOME OUTPUTS HERE ARE IN FLOAT
        return layernorm_diff, normalized_patch_embeddings_TSC11, patch_embeddings_TSC11_T, layernorm_diff_2, patch_embeddings_NC11_S_including_clf, layernorm_diff_3, patch_embeddings_NC11_S_including_clf_normalized, fc1_output, gelu, fc2_output, patch_embeddings, clf_token

    # 12 blocks of Temporal-Spatial Attention
    for idx in range(12):
        layernorm_diff, normalized_patch_embeddings_TSC11, patch_embeddings_TSC11_T, layernorm_diff_2, \
        patch_embeddings_NC11_S_including_clf, layernorm_diff_3, patch_embeddings_NC11_S_including_clf_normalized, fc1_output, gelu, fc2_output, \
        patch_embeddings_TSC11, clf_token = build_block(idx, patch_embeddings_TSC11, clf_token)


        if expose_layer_norm_entries:
            network.mark_output(layernorm_diff)
            network.mark_output(layernorm_diff_2)
            network.mark_output(layernorm_diff_3)
        network.mark_output(patch_embeddings_TSC11)

    shuffle_layer = network.add_shuffle(clf_token)
    shuffle_layer.reshape_dims = (1, 768, 1, 1)
    clf_token = shuffle_layer.get_output(0)

    weights = model_state_dict['model.norm.weight'].numpy().astype(np.float32)
    bias = model_state_dict['model.norm.bias'].numpy().astype(np.float32)
    _, _, _, clf_token_normalized = layer_norm(network, clf_token, weights,
                                               bias, 12, 0, axes=2)

    weights = model_state_dict['model.head.weight'].numpy().astype(np.float32)
    weights = trt.Weights(weights)
    bias = model_state_dict['model.head.bias'].numpy().astype(np.float32)
    bias = trt.Weights(bias)
    logits = network.add_fully_connected(clf_token_normalized, 400, weights, bias).get_output(0)

    shuffle_layer = network.add_shuffle(logits)
    shuffle_layer.reshape_dims = (1, 400)
    logits = shuffle_layer.get_output(0)

    softmax = network.add_softmax(logits)
    softmax.axes = 2
    probs = softmax.get_output(0)
    shuffle_layer = network.add_shuffle(probs)
    shuffle_layer.reshape_dims = (400,)
    probs = shuffle_layer.get_output(0)
    probs.name = 'probs'
    network.mark_output(probs)

    engine = builder.build_cuda_engine(network)
    print('Save the engine')
    with open(outputs_engine_path, "wb") as f:
        f.write(engine.serialize())

    print('Run evaluation on test data')
    inference = TRTInferenceModule(engine)
    test_inputs, expected_outputs = pickle.load(open(test_data_path, 'rb'))
    # test input BCTHW -> BTCHW
    test_inputs = test_inputs.transpose(0, 2, 1, 3, 4)
    outputs = inference.do_inference([test_inputs])
    probs_diff = np.linalg.norm(outputs[-1] - expected_outputs[0]) / np.linalg.norm(expected_outputs[0])
    print('Final probabilities difference:', probs_diff)
    for idx in range(12):
        expected_data = expected_outputs[1][idx]
        expected_patches = expected_data[0, 1:, :].reshape(196, 8, 768).transpose(1, 0, 2)
        if expose_layer_norm_entries:
            offset = 3
        else:
            offset = 0
        patches = outputs[idx * (offset + 1) + offset].reshape(8, 196, 768)
        diff = np.linalg.norm((expected_patches - patches)) / np.linalg.norm(expected_patches)
        print('Patch embedding diff after {0} S-T blocks: {1}'.format(idx, diff))
    pickle.dump(outputs, open(test_result_path, 'wb'))


if __name__ == '__main__':
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)

    model_state_path = arguments['<model_state_path>']
    test_data_path = arguments['<test_data_path>']
    outputs_engine_path = arguments['<outputs_engine_path>']
    test_result_path = arguments['<test_result_path>']
    fp16_mode = arguments['--fp16_mode']
    expose_ln_entries = arguments['--expose_ln_entries']

    conversions(model_state_path, test_data_path, outputs_engine_path, test_result_path, fp16_mode, expose_ln_entries)
