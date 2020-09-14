import os
import numpy as np
import elf
from elf.io import open_file
import h5py
import vigra
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
from pybdv import make_bdv
from pybdv import convert_to_bdv
import napari
from PIL import Image

import imageio

scale_factors = [[2,2,2], [2, 2, 2], [4, 4, 4]]
mode = 'mean'

#filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
raw_bdv = '/g/schwab/Viktoriia/src/source/upscaling_results/s4a2_bdv.h5'
# with open_file(filename_raw, 'r') as f:
#     full_shape = f['/t00000/s00/0/cells'].shape
#convert_to_bdv(filename_raw, '/t00000/s00/0/cells', raw_bdv, downscale_factors = scale_factors, downscale_mode = mode)
with h5py.File(raw_bdv, 'r') as f:
    raw = f['/t00000/s00/2/cells'][:, :, :]
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/updated_beta_0.6/'

mask_filename = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/MIBProcessing/s4a2_target_cell_mask.h5' #dataset data


#downsample results 
filename = 'result_stitching_file_final.h5'
filename_bdv = 'mito_segmentation_bdv.h5'
convert_to_bdv(res_path + filename, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)
with h5py.File(res_path + filename_bdv, 'r') as f:
      down_label = f['/t00000/s00/2/cells'][:,:,:]
shape = down_label.shape

#apply mask
with h5py.File(mask_filename, 'r') as f:
      mask = f['data'][:,:,:]
new_data = down_label * mask

filename_bdv_masked = 'mito_segmentation_bdv_mask.h5'
with h5py.File(res_path + filename_bdv_masked, 'w') as f:
    f.create_dataset('data', data = new_data, compression = 'gzip')
    #new_data = f['data'][:,:,:]


#filename_bdv = 'mito_segmentation_bdv_1.h5'
#convert_to_bdv(res_path + filename_bdv_masked, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)

#create color image for 3dViewer 
with h5py.File(res_path + filename_bdv, 'r') as f:
    #f.create_dataset('data', data = new_data, compression = 'gzip')
    data = f['/t00000/s00/2/cells'][:,:,:]

data, max_label, mapping  = vigra.analysis.relabelConsecutive(new_data.astype('uint64'), start_label=1, keep_zeros=True)

data = data.astype('uint8')
labels = np.unique(data)

max_label = np.max(data)
n = int(max_label ** (1/3)) + 1

R = np.linspace(1, 255, n)
G = np.linspace(1, 255, n)
B = np.linspace(1, 255, n)

i = 1

reds = {}
greens = {}
blues = {}
rgb = {0:[0,0,0]}
for r in R:
    for g in G:
        for b in B:
            reds[i] = r
            greens[i] = g
            blues[i] = b
            rgb[i] = [int(r), int(g), int(b)]
            i += 1
keys, values = zip(*rgb.items())
out = np.empty((max(keys) + 1,3), np.uint8)
out[list(keys)] = values

new_data_1 = out[data]
filename_tiff = 'mito_segmentation_rgb.tif'

imageio.volwrite(res_path + filename_tiff, new_data_1)
