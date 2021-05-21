# timesfromer_trtfp16_error
TimesFormer is the new Facebook's transformer based video model. 
https://github.com/facebookresearch/TimeSformer

This repository shows when converting timesformer to tensorRT in fp16, there are large discrepancies. 

We build network and convert in both fp32 and fp16, then compare 6 steps results for 12 self attention blocks:
- After T attention
- After S attention
- MLP - FC1
- MLP - GELU 
- MLP - FC2
- Final output


The main issue is that `gelu` magnifies errors accumulated from previous steps, and after 12 blocks of self attention, the error becomes too large

## Environment 
```
the environt is build based on the tensorrt docker in NGC
docker pull nvcr.io/nvidia/tensorrt:21.05-py3

TensorRT: 7.2.3.4
CUDA: 11.3
```

# To Run
```
# clone repo
git clone https://github.com/zmy1116/timesfromer_trtfp16_error

cd timesfromer_trtfp16_error

# download pretrained weights from TimeSformer official repo
wget https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth

# build 2 models in fp32 and fp16, get the outputs and do comparison
python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_inputs.p result_fp32.p

python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_inputs.p result_fp16.p --fp16_mode

# do comparison
python compare.py
```

The normalized difference between FP32 and FP16 entries should be printed on the screen 
```
    T-attention  S-attention       FC1      GELU       FC2    output
0      0.000429     0.001589  0.001332  0.003152  0.003271  0.002415
1      0.002342     0.002767  0.001660  0.004542  0.004563  0.003427
2      0.003424     0.003747  0.002070  0.005539  0.006106  0.004365
3      0.004315     0.004391  0.002267  0.005573  0.007846  0.005714
4      0.005700     0.006732  0.003855  0.010994  0.023288  0.014126
5      0.013844     0.013440  0.007358  0.029005  0.058700  0.049154
6      0.049353     0.054302  0.102258  0.473737  0.535258  0.336202
7      0.335790     0.335395  0.139182  0.372451  0.702222  0.348630
8      0.348025     0.349228  0.183254  0.399796  0.538561  0.356199
9      0.355522     0.361621  0.401468  0.669973  0.805795  0.383704
10     0.381401     0.397778  0.706607  0.966223  0.910731  0.400721
11     0.441942     0.557325  0.280049  1.161975  1.132105  0.569143
```



