import os
import pickle
import numpy as np
import napari
import elf
import elf.segmentation.workflows as elf_workflow
from elf.io import open_file
import h5py
import vigra

################################################
# training part
###############################################

path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
rf_save_path = './rf_dmv.pkl'


# rf_save_path = '/scratch/vgross/rf_mito.pkl'

def train():
    print("start edge training")
    raw_train = []
    mem_train = []
    dmv_train = []
    sv_train = []
    new_sv_labels_train = []

    for name in ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029']:
        # raw data
        filename = name + '/raw_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        raw_train.append(f['data'][:].astype(np.float32))

        # membrane prediction -- boundaries
        filename = name + '/mem_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        mem_train.append(f['data'][:].astype(np.float32))

        # ground truth for DMVs
        filename = name + '/results/raw_MITO.h5'
        f = open_file(path_train + filename, 'r')
        dmv_train.append(f['data'][:].astype(np.float32))

        # supervoxels
        filename = name + '/sv_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        sv_train.append(f['data'][:].astype('uint64'))  # (np.float32)

        newlabels_train, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_train[-1])
        new_sv_labels_train.append(newlabels_train)

    # do edge training
    rf = elf_workflow.edge_training(raw=raw_train, boundaries=mem_train, labels=dmv_train,
                                    use_2dws=False, watershed=new_sv_labels_train)
    print("edge training is done")
    return rf


if os.path.exists(rf_save_path):
    with open(rf_save_path, 'rb') as f:
        rf = pickle.load(f)
else:
    rf = train()
    with open(rf_save_path, 'wb') as f:
        pickle.dump(rf, f)

########################################################################
# segmentation part
########################################################################

path_sv = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'
filename_mem = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'
filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
# filename_mem = '/scratch/vgross/mem_pred_scale1.h5'
# filename_raw = '/scratch/vgross/amst_inv_clip_black_bg.h5'

# filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'
# filename_mem = '/g/schwab/Viktoriia/src/source/mem_crop.h5'
# filename_sv = '/g/schwab/Viktoriia/src/source/sv_crop.h5'

res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/'  # whole_dataset_dmv/  # + s4a2_mc_z_y_x_shape.h5
# res_path = '/scratch/vgross/upscaling_results_mito/updated_beta_0.6/'

with open_file(filename_raw, 'r') as f:
    shape = f['/t00000/s00/0/cells'].shape
# print("Full  shape:", shape)
nz, ny, nx = shape

################################################
# stitching sv
###############################################

stitched_sv_path = '/g/schwab/Viktoriia/src/source/s4a2_sv.h5'


def stitch_supervoxels():
    print('Start stitching supervoxels')

    with h5py.File(stitched_sv_path, mode='w') as f:
        f.create_dataset('data', shape=shape, compression='gzip', dtype='uint32')

    sv_file = h5py.File(stitched_sv_path, mode='a')
    sv_dataset = sv_file['data']

    # zerolabels = np.zeros(shape)
    # sv_dataset.write_direct(zerolabels, np.s_[:,:,:], np.s_[:,:,:])
    # zerolabels = None

    max_label = 0

    for z in range(0, shape[0], 256):
        for y in range(0, shape[1], 256):
            for x in range(0, shape[2], 256):
                filename = str(z) + '_' + str(y) + '_' + str(x) + '.h5'
                if os.path.exists(path_sv + filename):
                    print (z, y, x)
                    f = open_file(path_sv + filename, 'r')
                    if len(list(f.keys())) != 0:
                        if nz - z >= 128 and ny - y >= 128 and nx - x >= 128:
                            start_sv = (128 * int(z > 0), 128 * int(y > 0), 128 * int(x > 0))
                            end_sv = (min(384, shape[0] - z), min(384, shape[1] - y), min(384, shape[2] - x))
                            # part_sv = np.s_[start_sv[0]:end_sv[0], start_sv[1]:end_sv[1], start_sv[2]:end_sv[2]]
                            sv_current = f['data'][start_sv[0]:end_sv[0], start_sv[1]:end_sv[1],
                                         start_sv[2]:end_sv[2]].astype('uint32')
                            new_sv, new_max, mapping = vigra.analysis.relabelConsecutive(sv_current,
                                                                                         start_label=max_label + 1)
                            max_label = new_max
                            part = np.s_[z + start_sv[0]: z + end_sv[0], y + start_sv[1]: y + end_sv[1],
                                   x + start_sv[2]: x + end_sv[2]]
                            sv_dataset.write_direct(new_sv, np.s_[:, :, :], part)
                        # print(max_label)


if not (os.path.exists(stitched_sv_path)):
    stitch_supervoxels()
    print('done stitching supervoxels')

####################################################################
# segmentation by blocks
######################################################################

#filename_sv = '/scratch/vgross/s4a2_sv.h5'
filename_sv = '/g/schwab/Viktoriia/src/source/s4a2_sv.h5'

error_cubes = []

# mc_blocks = [256, 256, 256]
mc_blocks = [512, 512, 512]


def mc_segmentation(bb, mc_blocks, filename_raw, filename_mem, filename_sv):
    f = open_file(filename_raw, 'r')
    data_raw = f['/t00000/s00/0/cells'][bb].astype(np.float32)
    shape = data_raw.shape

    if np.min(data_raw) == np.max(data_raw):
        # print('no raw data ')
        # print(np.min(data_raw))
        return np.zeros(shape)

    f = open_file(filename_mem, 'r')
    data_mem = f['data'][bb].astype(np.float32).reshape(data_raw.shape)
    assert data_mem.shape == shape

    f = open_file(filename_sv, 'r')
    data_sv = f['data'][bb].astype('uint32')
    assert data_sv.shape == shape

    data_sv, maxlabel, mapping = vigra.analysis.relabelConsecutive(data_sv)

    # print("Final shape:", shape)

    # print('sv', np.min(data_sv), np.max(data_sv))

    if np.min(data_sv) == np.max(data_sv):
        # print('no superpixels')
        return np.zeros(shape)
    try:
        # run blockwise segmentation
        # print("Start segmentation")
        segmentation = elf_workflow.multicut_segmentation(raw=data_raw,
                                                          boundaries=data_mem,
                                                          rf=rf, use_2dws=False,
                                                          watershed=data_sv,
                                                          multicut_solver='blockwise-multicut',
                                                          solver_kwargs={'internal_solver': 'kernighan-lin',
                                                                         'block_shape': mc_blocks},
                                                          n_threads=16, beta=0.6)
        # print('segmentation is done')
        return segmentation
    except RuntimeError:
        error_cubes.append(bb)
        # print('runtime error in segmentation')
        return np.zeros(shape)


#################
# run segmentations for all the cubes

def run_segmentation_for_cubes(size_x=1024, size_y=1024, size_z=1024, overlap=256):
    # print("Start segmenting cubes")
    step_z, step_y, step_x = size_z - overlap, size_y - overlap, size_x - overlap
    for z in range(0, nz, step_z):
        for y in range(0, ny, step_y):
            for x in range(0, nx, step_x):
                # print(z, y, x)
                bb = np.s_[z: min(z + size_z, nz), y: min(y + size_y, ny), x: min(x + size_x, nx)]
                # filename = s4a2_mc_z_y_x_shape.h5
                filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' + str(size_z) + '_' + \
                                   str(size_y) + '_' + str(size_x) + '.h5'
                if not (os.path.exists(res_path + filename_results)):
                    segmentation = mc_segmentation(bb, mc_blocks, filename_raw, filename_mem, filename_sv)
                    f = open_file(res_path + filename_results, 'w')
                    f.create_dataset('data', data=segmentation, compression="gzip")


run_segmentation_for_cubes()
# print('done segmenting cubes')
# print(error_cubes)
#f = open('/scratch/vgross/errors.txt', 'w')
#f.write(str(error_cubes))
