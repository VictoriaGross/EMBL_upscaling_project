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

#import Image

# filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
# with open_file(filename_raw, 'r') as f:
#     full_shape = f['/t00000/s00/0/cells'].shape
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/'
#
# mask_filename = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/MIBProcessing/s4a2_target_cell_mask.h5' #dataset data
#
# scale_factors = [[2,2,2], [2, 2, 2], [4, 4, 4]]
# mode = 'nearest'
#
# filename = 'result_stitching_file_final.h5'
# filename_bdv = 'mito_segmentation_bdv.h5'
# convert_to_bdv(res_path + filename, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)
# with h5py.File(res_path + filename_bdv, 'r') as f:
#     down_label = f['/t00000/s00/2/cells'][:,:,:]
# shape = down_label.shape
#
# with h5py.File(mask_filename, 'r') as f:
#     mask = f['data'][:,:,:]
# new_data = down_label * mask

filename_bdv_masked = 'mito_segmentation_bdv_mask.h5'
with h5py.File(res_path + filename_bdv_masked, 'r') as f:
    #f.create_dataset('data', data = new_data, compression = 'gzip')
    data = f['data'][:,:,:]
#
# data, max_label, mapping  = vigra.analysis.relabelConsecutive(data.astype('uint64'), start_label=1, keep_zeros=True)
#
# #max_label = np.max(data)
# print(max_label)
# n = int(max_label ** (1/3)) + 1
# print(n)
# R = np.linspace(1, 255, n)
# G = np.linspace(1, 255, n)
# B = np.linspace(1, 255, n)
#
# i = 1
#
# reds = {}
# greens = {}
# blues = {}
# for r in R:
#     for g in G:
#         for b in B:
#             reds[i] = r
#             greens[i] = g
#             blues[i] = b
#             i += 1
#
# Rs = vigra.analysis.applyMapping(labels=data, mapping=reds, allow_incomplete_mapping=True)
# Gs = vigra.analysis.applyMapping(labels=data, mapping=greens, allow_incomplete_mapping=True)
# Bs = vigra.analysis.applyMapping(labels=data, mapping=blues, allow_incomplete_mapping=True)
#
# new_data = np.array([Rs, Gs, Bs])

filename = 'mito_segmentation_bdv_mask_rgb.h5'
with h5py.File(res_path + filename, 'r') as f:
    #f.create_dataset('data', data = new_data, compression = 'gzip')
    new_data = f['data'][:,:,:]
with h5py.File(res_path + filename, 'r') as f:
    f.create_dataset('data_8', data = new_data.astype('uint8'), compression = 'gzip')
# im = Image.fromarray(new_data)
# im.save(res_path + 'mito_rgb.tif')

