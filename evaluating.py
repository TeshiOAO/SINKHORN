import os
import math
import numpy as np
from tqdm import tqdm

data_shape = (64,64,64,)

def PSNR(raw, cmp):
    diff = raw - cmp
    diff = diff.flatten()
    rmse = math.sqrt(np.mean(diff ** 2))
    Max = max(raw.flatten())
    return 20*math.log10(Max/rmse)

def SSIM(raw, cmp):
    pass  

rawfilepath = ''
comparefilepath = ''
raw_data = np.fromfile(rawfilepath, dtype='float32').reshape(data_shape)
compare_data = np.fromfile(comparefilepath, dtype='float32').reshape(data_shape)

# Computing PSNR and SSIM
PSNR_Score = PSNR()
SSIM_Score = SSIM()