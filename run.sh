# download weights
#wget https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth

# generate results
# fp32 expose ln entries
nohup python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp32engine_expose_ln result_fp32_expose_ln.p --expose_ln_entries > trace_fp32_expose_ln_entries &

# fp32 without expose ln entries
nohup python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp32engine result_fp32.p > trace_fp32 &

# fp16 expose ln entries
nohup python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp16engine_expose_ln result_fp16_expose_ln.p --expose_ln_entries --fp16 > trace_fp16_expose_ln_entries &

# fp16 without expose ln entries
nohup python conversions.py TimeSformer_divST_8x32_224_K400.pyth test_data.p fp16engine result_fp16 --fp16 > trace_fp16 &