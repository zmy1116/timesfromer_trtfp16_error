# timesfromer_trtfp16_error
TimesFormer is the new Facebook's transformer based video model. 
https://github.com/facebookresearch/TimeSformer

This repository attempts to create TensorRT engine for TimeSformer. It shows when converting in fp16, there are large discrepancies. 

The network structure is fairly straightforward:
- for video of size 8x224x224, generate 8x14x14 patches of embeddings
- run 12 blocks of T-S self attention blocks
- use the final classification token as the feature to compute classification result 

Hence the following results are validated against known inputs-outputs:
- Final probability 
- Patches features after each of the 12 T-S blocks, 

So far we have following findings:
- The FP32 engine produces correct results
- The FP16 engine produces incorrect results without making any modifications, the error on patch features jump from 0.01 to 0.4 on 6th block 
- If we expose the difference tensor from layer norm as outputs, FP16 engine produces correct results


## Environment 
```
the environt is build based on the tensorrt docker in NGC
docker pull nvcr.io/nvidia/tensorrt:21.06-py3

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

# Build and test engine in FP16 with/without expose layernorm diffs as outputs
python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp16engine_expose_ln result_fp16_expose_ln.p --expose_ln_entries --fp16
python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp16engine result_fp16 --fp16


# Build and test engine in FP32 with/without expose layernrom diffs as outputs 
python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp32engine_expose_ln result_fp32_expose_ln.p --expose_ln_entries
python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp32engine result_fp32
```


I have included printed traces/outputs for 4 cases in the `traces` folder. 

Results on FP16 without exposing layernorm diffs looks like following, notice sudden error jump at 6th block
```
Run evaluation on test data
Final probabilities difference: 1.0182867
Patch embedding diff after 0 S-T blocks: 0.004299811087548733
Patch embedding diff after 1 S-T blocks: 0.00536126596853137
Patch embedding diff after 2 S-T blocks: 0.00613724160939455
Patch embedding diff after 3 S-T blocks: 0.00694211944937706
Patch embedding diff after 4 S-T blocks: 0.009320137090981007
Patch embedding diff after 5 S-T blocks: 0.013058185577392578
Patch embedding diff after 6 S-T blocks: 0.4091333746910095
Patch embedding diff after 7 S-T blocks: 0.430844247341156
Patch embedding diff after 8 S-T blocks: 0.44067439436912537
Patch embedding diff after 9 S-T blocks: 0.46074366569519043
Patch embedding diff after 10 S-T blocks: 0.45822423696517944
Patch embedding diff after 11 S-T blocks: 0.5575547814369202
```

Results on FP16 with exposing layernorm diffs looks like following 
```
Final probabilities difference: 0.0033378536
Patch embedding diff after 0 S-T blocks: 0.0042906636372208595
Patch embedding diff after 1 S-T blocks: 0.005335847847163677
Patch embedding diff after 2 S-T blocks: 0.0060969325713813305
Patch embedding diff after 3 S-T blocks: 0.006786249577999115
Patch embedding diff after 4 S-T blocks: 0.00906979851424694
Patch embedding diff after 5 S-T blocks: 0.014810287393629551
Patch embedding diff after 6 S-T blocks: 0.015742896124720573
Patch embedding diff after 7 S-T blocks: 0.015805872157216072
Patch embedding diff after 8 S-T blocks: 0.015810402110219002
Patch embedding diff after 9 S-T blocks: 0.01605822518467903
Patch embedding diff after 10 S-T blocks: 0.016058534383773804
Patch embedding diff after 11 S-T blocks: 0.01618446595966816

```
