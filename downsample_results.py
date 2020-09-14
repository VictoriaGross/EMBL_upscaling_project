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

filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
raw_bdv = '/g/schwab/Viktoriia/src/source/upscaling_results/s4a2_bdv.h5'
# with open_file(filename_raw, 'r') as f:
#     full_shape = f['/t00000/s00/0/cells'].shape
#convert_to_bdv(filename_raw, '/t00000/s00/0/cells', raw_bdv, downscale_factors = scale_factors, downscale_mode = mode)
with h5py.File(raw_bdv, 'r') as f:
    raw = f['/t00000/s00/2/cells'][:, :, :]
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/updated_beta_0.6/'
#
#mask_filename = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/MIBProcessing/s4a2_target_cell_mask.h5' #dataset data
#

#
filename = 'result_stitching_file_final.h5'
filename_bdv = 'mito_segmentation_bdv.h5'
#convert_to_bdv(res_path + filename, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)
with h5py.File(res_path + filename_bdv, 'r') as f:
      down_label = f['/t00000/s00/2/cells'][:,:,:]
shape = down_label.shape

x_lines  = np.arange(0, shape[2], 768 // 4)
x_lines_1  = np.arange(256, shape[2], 768 // 4)
y_lines  = np.arange(0, shape[1], 768 // 4)
y_lines_1  = np.arange(256, shape[1], 768 // 4)

nx = x_lines.size
nx_1 = x_lines_1.size
ny = y_lines.size
ny_1 = y_lines_1.size

vertices = []

# for i in range (nx):
#     for z in range(shape[0]):
#         vertices.append(np.array([(z, 0, x_lines[i]), (z, shape[1] - 1, x_lines[i])]))
# for i in range (nx_1):
#     for z in range(shape[0]):
#         vertices.append(np.array([(z, 0, x_lines_1[i]), (z, shape[1] - 1, x_lines_1[i])])) #, (shape[0] - 1, 0, x_lines[i]), (shape[0] - 1, shape[1] - 1, x_lines[i])]))
# for i in range(ny):
#     for z in range(shape[0]):
#         vertices.append(np.array([(z, y_lines[i], 0), (z, y_lines[i], shape[2])])) #, (shape[0] - 1, 0, x_lines[i]), (shape[0] - 1, shape[1] - 1, x_lines[i])]))
# for i in range(ny_1):
#     for z in range(shape[0]):
#         vertices.append(np.array([(z, y_lines_1[i], 0), (z, y_lines_1[i], shape[2])])) #, (shape[0] - 1, 0, x_lines[i]), (shape[0] - 1, shape[1] - 1, x_lines[i])]))

reg_1_z0, reg_1_y0, reg_1_x0 = 1073 // 4, 1944 // 4, 1344 // 4
reg_1_z1, reg_1_y1, reg_1_x1 = 1585 // 4, 2456 // 4, 1856 // 4
reg_2_z0, reg_2_y0, reg_2_x0 = 1624 // 4, 2181 // 4, 1646 // 4
reg_2_z1, reg_2_y1, reg_2_x1 = 2136 // 4, 2693 // 4, 2158 // 4
reg_3_z0, reg_3_y0, reg_3_x0 = 2188 // 4, 311 // 4, 1619 // 4
reg_3_z1, reg_3_y1, reg_3_x1 = 2700 // 4, 823 // 4, 2131 // 4
reg_4_z0, reg_4_y0, reg_4_x0 = 2655 // 4, 1610 // 4, 2414 // 4
reg_4_z1, reg_4_y1, reg_4_x1 = 3167 // 4, 2122 // 4, 2926 // 4

for z in range (reg_1_z0, reg_1_z1):
    vertices.append(np.array([
        (z, reg_1_y0, reg_1_x0), (z, reg_1_y1, reg_1_x0), (z, reg_1_y1, reg_1_x1), (z, reg_1_y0, reg_1_x1)
    ]))
for z in range (reg_2_z0, reg_2_z1):
    vertices.append(np.array([
        (z, reg_2_y0, reg_2_x0), (z, reg_2_y1, reg_2_x0), (z, reg_2_y1, reg_2_x1), (z, reg_2_y0, reg_2_x1)
    ]))
for z in range (reg_3_z0, reg_3_z1):
    vertices.append(np.array([
        (z, reg_3_y0, reg_3_x0), (z, reg_3_y1, reg_3_x0), (z, reg_3_y1, reg_3_x1), (z, reg_3_y0, reg_3_x1)
    ]))
for z in range (reg_4_z0, reg_4_z1):
    vertices.append(np.array([
        (z, reg_4_y0, reg_4_x0), (z, reg_4_y1, reg_4_x0), (z, reg_4_y1, reg_4_x1), (z, reg_4_y0, reg_4_x1)
    ]))


with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(raw, name= 'raw')
    viewer.add_labels(down_label, name = 'MITO')
    viewer.add_shapes(vertices, shape_type = 'rectangle', edge_color = 'red' )

# #
# with h5py.File(mask_filename, 'r') as f:
#       mask = f['data'][:,:,:]
# new_data = down_label * mask
#
# filename_bdv_masked = 'mito_segmentation_bdv_mask.h5'
# with h5py.File(res_path + filename_bdv_masked, 'w') as f:
#     f.create_dataset('data', data = new_data, compression = 'gzip')
#     #new_data = f['data'][:,:,:]

# scale_factors = [2,2,2]
# mode = 'nearest'
#filename_bdv = 'mito_segmentation_bdv_1.h5'
#convert_to_bdv(res_path + filename_bdv_masked, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)

# with h5py.File(res_path + filename_bdv, 'r') as f:
#     #f.create_dataset('data', data = new_data, compression = 'gzip')
#     data = f['/t00000/s00/2/cells'][:,:,:]

#data, max_label, mapping  = vigra.analysis.relabelConsecutive(new_data.astype('uint64'), start_label=1, keep_zeros=True)

#data = data.astype('uint8')
# labels = np.unique(data)
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
# rgb = {0:[0,0,0]}
# for r in R:
#     for g in G:
#         for b in B:
#             reds[i] = r
#             greens[i] = g
#             blues[i] = b
#             rgb[i] = [int(r), int(g), int(b)]
#             i += 1
# keys, values = zip(*rgb.items())
#
# print(keys)
# print(values)
#
# out = np.empty((max(keys) + 1,3), np.uint8)
# out[list(keys)] = values
#
# new_data_1 = out[data]
#
# print(new_data_1.shape)
# # Rs = (vigra.analysis.applyMapping(labels=data, mapping=reds, allow_incomplete_mapping=True)).astype('uint8')
# # Gs = (vigra.analysis.applyMapping(labels=data, mapping=greens, allow_incomplete_mapping=True)).astype('uint8')
# # Bs = (vigra.analysis.applyMapping(labels=data, mapping=blues, allow_incomplete_mapping=True)).astype('uint8')
# # #new_data_1 = vigra.analysis.applyMapping(labels=data, mapping=rgb, allow_incomplete_mapping=True)
# #
# # #
# # new_data_1 = (Rs, Gs, Bs)
# # print(new_data_1.shape)
# filename_tiff = 'mito_segmentation_rgb.tif'
#
# imageio.volwrite(res_path + filename_tiff, new_data_1)
# #

# with h5py.File(res_path + filename, 'w') as f:
#     f.create_dataset('data', data = new_data.astype('uint8'), compression = 'gzip', dtype = 'u2')
# #     new_data = f['data'][:,: ,:,:]
    #f.create_dataset('data_8', data = new_data.astype('uint8'), compression = 'gzip')
# im = Image.fromarray(new_data)
# im.save(res_path + 'mito_rgb.tif')
# print(new_data.shape)
# scale_factors = [1,2,2,2]
# mode = 'nearest'
#
# #filename = 'result_stitching_file_final.h5'
# filename_bdv = 'mito_segmentation_rgb_bdv.h5'
# convert_to_bdv(res_path + filename, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)
# #with h5py.File(res_path + filename_bdv, 'r') as f:
# #    down_label = f['/t00000/s00/2/cells'][:,:,:]
# # shape = down_label.shape
# #
