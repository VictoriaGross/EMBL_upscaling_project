from run_stitching import *
import os
import numpy as np
import h5py
import vigra
from concurrent.futures import ThreadPoolExecutor


path_sv = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'
filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
sv_pattern = '{}_{}_{}.h5'
res_path = '/g/schwab/Viktoriia/src/source/sv_stitching/'

run_stitching(
    filename_raw=filename_raw, dataset_raw='/t00000/s00/0/cells',
    output_folder = res_path,
    input_folder = path_sv,
    input_file_pattern = sv_pattern,
    step=(256, 256, 256), n_threads=4, overlap=256
)