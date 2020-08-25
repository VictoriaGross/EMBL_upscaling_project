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
rf_save_path = './rf_dmv.pkl'

#rf_save_path = '/scratch/vgross/rf_dmv.pkl'

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


########################################################################
# segmentation part
########################################################################

#path_sv = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'
filename_mem = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'
filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
#filename_mem = '/scratch/vgross/mem_pred_scale1.h5'
#filename_raw = '/scratch/vgross/amst_inv_clip_black_bg.h5'

#filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'
#filename_mem = '/g/schwab/Viktoriia/src/source/mem_crop.h5'
#filename_sv = '/g/schwab/Viktoriia/src/source/sv_crop.h5'

res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_dmv/'  # + s4a2_mc_z_y_x_shape.h5
#res_path = '/scratch/vgross/upscaling_results_dmv/'

with open_file(filename_raw, 'r') as f:
    shape = f['/t00000/s00/0/cells'].shape
#print("Full  shape:", shape)
nz, ny, nx = shape

################################################
#stitching sv
###############################################

#stitched_sv_path = '/g/schwab/Viktoriia/src/source/s4a2_sv.h5'
"""
def stitch_supervoxels():
	print('Start stitching supervoxels')

	with h5py.File(stitched_sv_path, mode='w') as f:
		f.create_dataset('data', shape=shape, compression='gzip', dtype='uint32')

	sv_file = h5py.File(stitched_sv_path, mode='a')
	sv_dataset = sv_file['data']

	#zerolabels = np.zeros(shape)
	#sv_dataset.write_direct(zerolabels, np.s_[:,:,:], np.s_[:,:,:])
	#zerolabels = None

	max_label = 0

	for z in range (0, shape[0], 256):
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
							#part_sv = np.s_[start_sv[0]:end_sv[0], start_sv[1]:end_sv[1], start_sv[2]:end_sv[2]]
							sv_current = f['data'][start_sv[0]:end_sv[0], start_sv[1]:end_sv[1], start_sv[2]:end_sv[2]].astype('uint32')
							new_sv, new_max, mapping = vigra.analysis.relabelConsecutive(sv_current, start_label = max_label + 1)
							max_label = new_max
							part = np.s_[z + start_sv[0]: z + end_sv[0], y + start_sv[1] : y + end_sv[1], x + start_sv[2] : x + end_sv[2]]
							sv_dataset.write_direct(new_sv, np.s_[:,:,:],  part)
							print(max_label)


if not(os.path.exists(stitched_sv_path)):
	stitch_supervoxels()
	print('done stitching supervoxels')
"""
####################################################################
#segmentation by blocks
######################################################################

#filename_sv = '/scratch/vgross/s4a2_sv.h5'
filename_sv = './s4a2_sv.h5'

error_cubes = []

mc_blocks = [256, 256, 256]

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

	#print("Final shape:", shape)

	#print('sv', np.min(data_sv), np.max(data_sv))

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
                                                      n_threads=4)
		print('segmentation is done')
		return segmentation
	except RuntimeError:
		error_cubes.append(bb)
		#print('runtime error in segmentation')
		return np.zeros(shape)

#################
# run segmentations for all the cubes

def run_segmentation_for_cubes(size_x = 1024, size_y = 1024, size_z = 1024, overlap = 256):
	#print("Start segmenting cubes")
	step_z, step_y, step_x = size_z - overlap, size_y - overlap, size_x - overlap
	for z in range(0, nz, step_z):
		for y in range(0, ny, step_y):
			for x in range(0, nx, step_x):
				print(z, y, x)
				bb = np.s_[z: min(z + size_z, nz), y: min(y + size_y, ny), x: min(x + size_x, nx)]
				# filename = s4a2_mc_z_y_x_shape.h5
				filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' +  str(size_z)+ '_' + \
									   str(size_y) + '_' +  str(size_x) + '.h5'
				if not(os.path.exists(res_path + filename_results)):
					segmentation = mc_segmentation(bb, mc_blocks, filename_raw, filename_mem, filename_sv)
					f = open_file(res_path + filename_results, 'w')
					f.create_dataset('data', data=segmentation, compression="gzip")

run_segmentation_for_cubes()
#print('done segmenting cubes')
#print(error_cubes)
f = open('/scratch/vgross/errors.txt', 'w')
f.write(str(error_cubes))

################################################
# stitching part
###############################################
"""
print("start stitching part")

with h5py.File(res_path + 'result_stitching_file_dmv.h5', mode = 'w') as f:
	f.create_dataset('data', shape = shape, compression = 'gzip', dtype = 'uint32')

result_file = h5py.File(res_path + 'result_stitching_file_dmv.h5', mode='a')
result_dataset = result_file['data']

maxlabel = 0

def stitching_row (start_index = (0,0,0), row_count = 0, step = 512, shape = (1024, 2304, 1792), maxlabel = maxlabel):
	name = 'data_row' + str(row_count)
	result_file.create_dataset(name, shape = shape, compression = 'gzip', dtype = 'uint32')
	result_dataset_row = result_file[name]

	z = start_index[0]
	y = start_index[1]
	size_cube = step + 256
	current_shape = (0, 0, 0)
	for x in range (start_index[2], shape[2], step):
		if (x + y + z > 0) and (nx - x > 256 and ny - y > 256 and nz - z > 256):
			bb = np.s_[z: min(z + size_cube, nz), y: min(y + size_cube, ny), x: min(x + size_cube, nx)]
			#print(z, y, x)
			#print(bb)
			filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' + str(size_cube) + '_' + str(size_cube) + '_' + str(size_cube) + '.h5'
			f = open_file(res_path + filename_results, 'r')
			current_label = f['data'][:, :, :]
			if current_shape == (0,0,0):
				current_shape = current_label.shape
				part = np.s_[: current_shape[0], : current_shape[1], : current_shape[2]]

				current_label, new_maxlabel, mapping = vigra.analysis.relabelConsecutive(current_label.astype('uint64'), start_label=maxlabel + 1,
																				keep_zeros=True)
				current_label = vigra.analysis.applyMapping(current_label, mapping={maxlabel + 1: 0}, allow_incomplete_mapping=True)

				maxlabel = new_maxlabel
				result_dataset_row.write_direct(current_label, np.s_[:, :, :],
											np.s_[: current_shape[0], : current_shape[1], : current_shape[2]])

			else:
				maxlabel, current_shape = stitching_function(result_dataset_row, current_label, current_shape, current_maxlabel=maxlabel, start_index_2=(0, 0, x))
			#print('row stitching function', current_shape, maxlabel)
	return current_shape, maxlabel

def stitching_function(result_dataset, new_region, current_shape, current_maxlabel=0, start_index_2 = None):

	#print('start stitching')

	new_region, max_label, mapping = vigra.analysis.relabelConsecutive(new_region.astype('uint64'), start_label=current_maxlabel + 1, keep_zeros=True)
	new_region = vigra.analysis.applyMapping(labels=new_region, mapping={current_maxlabel + 1: 0}, allow_incomplete_mapping=True)

	current_maxlabel = max_label
	#print('new max label', max_label)

	shape2 = new_region.shape
	#print ('shape of new region', shape2)

	result_data = np.zeros(shape2)
	index = np.s_[start_index_2[0] : start_index_2[0] + shape2[0],
			start_index_2[1] : start_index_2[1] + shape2[1],
			start_index_2[2] : start_index_2[2] + shape2[2]]
	result_dataset.read_direct(result_data, index, np.s_[:,:,:])
	#print('done reading')

	# get shapes
	shape1 = current_shape
	shape2 = new_region.shape

	shape = np.zeros(3).astype('int')
	for i in range(3):
		shape[i] = max(start_index_2[i] + shape2[i], shape1[i])

	shape = tuple(shape)

	#print('new shape', shape)

	# non-overlapping regions
	print('start non-overlap')
	for i in range(3):
		if start_index_2[i] + shape2[i] > shape1[i]:
			start = [0, 0, 0]
			start[i] = shape1[i]
			part = np.s_[start[0] : start_index_2[0] + shape2[0], start[1] : start_index_2[1] + shape2[1], start[2] : start_index_2[2] + shape2[2]]
			start[i] = shape1[i] - start_index_2[i]
			part1 = np.s_[start[0] : start_index_2[0] + shape2[0], start[1] : start_index_2[1] + shape2[1], start[2] : start_index_2[2] + shape2[2]]
			#result_dataset.write_direct(new_region, source_sel = part1, dest_sel = part)
			#print(part1)
			result_data[part1] = np.copy(new_region[part1])
	#print('done with non-overlapping')

	# overlapping region
	start_in1 = np.array(start_index_2)
	end_in1 = np.array((min(start_in1[0] + shape2[0], shape1[0]),
							min(start_in1[1] + shape2[1], shape1[1]),
							min(start_in1[2] + shape2[2], shape1[2])))

	part_in1 = np.s_[start_in1[0]: end_in1[0], start_in1[1]: end_in1[1], start_in1[2]: end_in1[2]]

	start_in2 = np.array((0,0,0))
	end_in2 = np.array((min(shape2[0], shape1[0] - start_in1[0]),
							min(shape2[1], shape1[1] - start_in1[1]),
							min(shape2[2], shape1[2] - start_in1[2])))
	part_in2 = np.s_[start_in2[0]: end_in2[0], start_in2[1]: end_in2[1], start_in2[2]: end_in2[2]]

	part_in1 = part_in2

	start_new = np.array(start_index_2)
	end_new = np.array((min(start_new[0] + shape2[0], shape1[0]),
							min(start_new[1] + shape2[1], shape1[1]),
							min(start_new[2] + shape2[2], shape1[2])))
	part_new = np.s_[start_new[0]: end_new[0], start_new[1]: end_new[1], start_new[2]: end_new[2]]

	labels_overlap1 = np.unique(result_data[part_in1])
	labels_overlap2 = np.unique(new_region[part_in2])

	#print('labels_overlap_1', labels_overlap1)
	#print('labels overlap 2', labels_overlap2)

	#print ('part in 1', part_in1)
	#print ('part in 2', part_in2)
	#print ('part in overall ', part_new)

	mapping_overlap = {}
	sizes_overlap1 = {}
	sizes_overlap2 = {}
	sizes_intercept = {}

	for i in labels_overlap1:
		if i != 0:
			sizes_overlap1[i] = np.count_nonzero(result_data[part_in1] == i)

	for i in labels_overlap2:
		if i != 0:
			sizes_overlap2[i] = np.count_nonzero(new_region[part_in2] == i)

	#print('done with sizes of each object')

	shape_overlap = result_data[part_in1].shape

	for z in range(shape_overlap[0]):
		for y in range(shape_overlap[1]):
			for x in range(shape_overlap[2]):
				l1 = result_data[part_in1][z, y, x]
				l2 = new_region[part_in2][z, y, x]
				if l1 != 0 and l2 != 0:
					if l2 in mapping_overlap.keys() and not (l1 in mapping_overlap[l2]):
						np.append(mapping_overlap[l2], l1)
						np.append(sizes_intercept[l2], 1)
					elif l2 in mapping_overlap.keys() and l1 in mapping_overlap[l2]:
						sizes_intercept[l2][np.where(mapping_overlap[l2] == l1)[0][0]] += 1
					elif not (l2 in mapping_overlap.keys()):
						mapping_overlap[l2] = np.array([l1])
						sizes_intercept[l2] = np.array([1])
	#print(mapping_overlap)
	#print('done mapping')

	iou = {}
	for l2 in mapping_overlap.keys():
		n = mapping_overlap[l2].size
		iou[l2] = np.zeros(n)
		for i in range(n):
			l1 = mapping_overlap[l2][i]
			iou[l2][i] = sizes_intercept[l2][i] / (sizes_overlap1[l1] + sizes_overlap2[l2] - sizes_intercept[l2][i])
	#print('done iou')

	for z in range(shape_overlap[0]):
		for y in range(shape_overlap[1]):
			for x in range(shape_overlap[2]):
				l1 = result_data[part_in1][z, y, x]
				l2 = new_region[part_in2][z, y, x]
				if l1 == 0 and l2 != 0:
					if l2 in mapping_overlap.keys():
						n = mapping_overlap[l2].size
						i = 0
						if n > 1:
							i = np.argmax(iou[l2])
						if iou[l2][i] >= 0.8:
							result_data[part_in2][z, y, x] = mapping_overlap[l2][i]
						else:
							result_data[part_in2][z, y, x] = l2
					else:
						result_data[part_in2][z, y, x] = l2
	#print('done overlap')

	for l2 in mapping_overlap.keys():
		n = mapping_overlap[l2].size
		i = 0
		if n > 1:
			i = np.argmax(iou[l2])
		if iou[l2][i] >= 0.8:
			result_data = np.where(result_data == l2, mapping_overlap[l2][i], result_data)
	#print('done stitching')
	result_dataset.write_direct(result_data, np.s_[:, :, :], index)
	return (current_maxlabel + 1, shape)

def stitching_column(step_x = 768, step_y = 768, z = 0, maxlabel = 0, column_index = 0, total_shape = (1024, 2304, 1792), max_row_index = 0, n_workers = 1):
	row_index = max_row_index
	shapes_of_rows = list()
	maxlabels = []
	if n_workers == 1:
		for y in range(0, ny, step_y):
			row_shape, maxlabel = stitching_row(start_index=(z, y, 0), row_count=row_index, step=step_x,
												shape=total_shape, maxlabel=maxlabel, axis=2)
			row_index += 1
			shapes_of_rows.append(row_shape)
			print('column stitching, row_index, maxlabel, y, z ', row_index, maxlabel, y, z)
	else:
		y_positions = np.arange(0, ny, step_y)
		row_indexes = np.arange(row_index, row_index + len(y_positions))
		variables = []
		for i in range(len(y_positions)):
			variables.append((y_positions[i], row_indexes[i]))
		with ThreadPoolExecutor(max_workers=n_workers) as tpe:
			# submit the tasks
			tasks = [
				tpe.submit(stitching_row, start_index=(z, y, 0), row_count=row_index, step=step_x,
						   shape=total_shape, maxlabel=maxlabel, axis=2)
				for y, row_index in variables
			]
			# get results
			results = [task.result() for task in tasks]
	print ('threads results ', results)
	for res in results:
		shapes_of_rows.append(res[0])
		maxlabels.append(res[1])
	maxlabel = maxlabels[0]
	# row_index = len(list(result_file.keys())) - 1
	y = 0
	x = 0
	current_shape = (0, 0, 0)

	name = 'data_column' + str(column_index)

	result_file.create_dataset(name, shape=total_shape, compression='gzip', dtype='uint64')
	result_dataset_column = result_file[name]

	for i in range(row_index - max_row_index):
		print(y)
		name = 'data_row' + str(i + max_row_index)
		current_label = result_file[name][:shapes_of_rows[i][0], :shapes_of_rows[i][1], :shapes_of_rows[i][2]]
		print(current_label.shape)
		if current_shape == (0, 0, 0):
			current_shape = current_label.shape
			part = np.s_[: current_shape[0], y: current_shape[1], : current_shape[2]]
			result_dataset_column.write_direct(current_label, np.s_[:, :, :], part)
		else:
			maxlabel, current_shape = stitching_function(result_dataset_column, current_label, current_shape,
														 current_maxlabel=maxlabel,
														 start_index_2=(z, y, x))
		y += step_y
		print(current_shape, maxlabel)
	return current_shape, maxlabel, row_index + 1


def stitch_total(step_x=768, step_y=768, step_z=768, maxlabel=0, total_shape=(1024, 2304, 1792), n_workers=1):
	print('start stitch total')
	row_index = 0
	column_index = 0
	shapes_of_columns = list()
	# threads = list()
	for z in range(0, nz, step_z):
		print('z = ', z)

		column_shape, maxlabel, row_index = stitching_column(step_x=step_x, step_y=step_y, z=z, maxlabel=maxlabel,
															 column_index=column_index, total_shape=total_shape,
															 max_row_index=row_index, n_workers=n_workers)

		column_index += 1
		shapes_of_columns.append(column_shape)
		print('column_shape = ', column_shape)

	# row_index = len(list(result_file.keys())) - 1
	z = 0
	y = 0
	x = 0
	current_shape = (0, 0, 0)
	# result_dataset = result_file['data']
	print('column index ', column_index)
	for i in range(column_index):
		name = 'data_column' + str(i)
		current_label = result_file[name][:shapes_of_columns[i][0], : shapes_of_columns[i][1], :shapes_of_columns[i][2]]
		if current_shape == (0, 0, 0):
			current_shape = current_label.shape
			part = np.s_[: current_shape[0], y: current_shape[1], : current_shape[2]]
			result_dataset.write_direct(current_label, np.s_[:, :, :], part)
		else:
			maxlabel, current_shape = stitching_function(result_dataset, current_label, current_shape,
														 current_maxlabel=maxlabel,
														 start_index_2=(z, y, x))
		z += step_z
		print(current_shape, maxlabel)
	print('done')
	return current_shape, maxlabel

current_shape, maxlabel = stitch_total(step_x = 768, step_y = 768, step_z = 768, maxlabel = 0, total_shape = shape, n_workers = 4)
print('done stitching')

with napari.gui_qt():
	viewer = napari.Viewer()
	viewer.add_image(raw, name='raw')
	viewer.add_labels(result_dataset[:,:,:], name='stitched')

"""