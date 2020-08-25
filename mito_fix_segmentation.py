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

#path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
rf_save_path = './rf_mito.pkl'

#rf_save_path = '/scratch/vgross/rf_mito.pkl'

def train():
    print("start edge training")
    raw_train = []
    mem_train = []
    dmv_train = []
    sv_train = []
    new_sv_labels_train = []

    for name in ['s4a2_t010', 's4a2_t012']:
        # raw data
        filename = name + '/raw_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        raw_train.append(f['data'][:].astype(np.float32))

        # membrane prediction -- boundaries
        filename = name + '/mem_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        mem_train.append(f['data'][:].astype(np.float32))

        # ground truth for DMVs
        filename = name + '/results/raw_DMV.h5'
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


#filename_mem = '/scratch/vgross/mem_pred_scale1.h5'
#filename_raw = '/scratch/vgross/amst_inv_clip_black_bg.h5'
#filename_sv = '/scratch/vgross/s4a2_sv.h5'
#res_path = '/scratch/vgross/upscaling_results_mito/update/'
filename_mem = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'
filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
filename_sv = './s4a2_sv.h5'
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/updated_beta_0.6/'

with open_file(filename_raw, 'r') as f:
    shape = f['/t00000/s00/0/cells'].shape
nz, ny, nx = shape

error_cubes = []

mc_blocks = [512, 512, 512]
#mc_blocks = [256, 256, 256]
def mc_segmentation(bb, mc_blocks, filename_raw, filename_mem, filename_sv):
	f = open_file(filename_raw, 'r')
	data_raw = f['/t00000/s00/0/cells'][bb].astype(np.float32)
	shape = data_raw.shape

	if np.min(data_raw) == np.max(data_raw):
		#print('no raw data ')
		#print(np.min(data_raw))
		return np.zeros(shape)

	f = open_file(filename_mem, 'r')
	data_mem = f['data'][bb].astype(np.float32).reshape(data_raw.shape)
	assert data_mem.shape == shape

	f = open_file(filename_sv, 'r')
	data_sv = f['data'][bb].astype('uint32')
	assert data_sv.shape == shape

	data_sv, maxlabel, mapping = vigra.analysis.relabelConsecutive(data_sv)

	if np.min(data_sv) == np.max(data_sv):
		#print('no superpixels')
		return np.zeros(shape)
	try:
		# run blockwise segmentation
		print("Start segmentation")
		segmentation = elf_workflow.multicut_segmentation(raw=data_raw,
                                                      boundaries=data_mem,
                                                      rf=rf, use_2dws=False,
                                                      watershed=data_sv,
                                                      multicut_solver='blockwise-multicut',
                                                      solver_kwargs={'internal_solver': 'kernighan-lin',
                                                                     'block_shape': mc_blocks},
                                                      n_threads=4, beta = 0.6)
		print('segmentation is done')
		return segmentation
	except RuntimeError:
		error_cubes.append(bb)
		#print('runtime error in segmentation')
		return np.zeros(shape)


bb = np.s_[3840 : , 2304 : 2304 + 1024 , : 1024]

segmentation = mc_segmentation(bb, mc_blocks, filename_raw, filename_mem, filename_sv)

filename_results = 's4a2_mc_3840_2304_0_1024_1024_1024.h5'
f = open_file(res_path + filename_results, 'w')
f.create_dataset('data', data=segmentation, compression="gzip")
