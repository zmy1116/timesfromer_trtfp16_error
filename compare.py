import numpy as np
import pandas as pd
import pickle

results_fp16 = pickle.load(open('result_fp16.p', 'rb'))
results_fp32 = pickle.load(open('result_fp32.p', 'rb'))
diffs = []

for idx in range(12):

    r = []
    for idx2 in range(6):
        r_fp32 = results_fp32[idx * 6 + idx2]
        r_fp16 = results_fp16[idx * 6 + idx2]
        diff = np.linalg.norm(r_fp32 - r_fp16)

        diff_norm = diff / np.linalg.norm(r_fp32)
        r.append(diff_norm)

    diffs.append(r)

df = pd.DataFrame(diffs, columns=['T-attention', 'S-attention','FC1', 'GELU', 'FC2', 'output'])
print(df)