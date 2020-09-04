import os
import numpy as np
import h5py
import vigra
from concurrent.futures import ThreadPoolExecutor
import sys

maxlabel = 0

def stitching_row(start_index=(0, 0, 0), row_count=0, step=512, total_shape=(1024, 2304, 1792), maxlabel=maxlabel, axis=2, res_path = None , overlap = 256):
    name = 'result_stitching_row_' + str(row_count) + '.h5'
    if os.path.exists(res_path + name) :
        current_shape = (min(total_shape[0] - start_index[0], step + overlap), min(total_shape[1] - start_index[1], step + overlap), total_shape[2])
        f = h5py.File(res_path + name, 'a')
        bb = np.s_[:current_shape[0], : current_shape[1], : current_shape[2]]
        maxlabel = np.max(f['data'][bb])
        return current_shape, maxlabel

    with h5py.File(res_path + name, mode='w') as f:
        f.create_dataset('data', shape=total_shape, compression='gzip', dtype='uint64')

    result_file_row = h5py.File(res_path + name, mode='a')
    result_dataset_row = result_file_row['data']

    z = start_index[0]
    y = start_index[1]
    x = start_index[2]
    coordinate = list(start_index)
    end = False

    nz, ny, nx = total_shape
    size_cube = step + 256
    current_shape = (0, 0, 0)
    while not end:
        filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' + '1024_1024_1024.h5'
        if os.path.exists(res_path + filename_results):
            f = h5py.File(res_path + filename_results, 'r')
            current_label = f['data'][:, :, :]
            if current_shape == (0, 0, 0):
                current_shape = current_label.shape
                part = np.s_[ :    current_shape[0], :  current_shape[1],  : current_shape[2]]
                current_label, new_maxlabel, mapping = vigra.analysis.relabelConsecutive(current_label.astype('uint64'),
                                                                                         start_label=maxlabel + 1,
                                                                                         keep_zeros=True)
                current_label = vigra.analysis.applyMapping(current_label, mapping={maxlabel + 1: 0},
                                                            allow_incomplete_mapping=True)
                maxlabel = new_maxlabel
                result_dataset_row.write_direct(current_label, np.s_[:, :, :],part)
            else:
                maxlabel, current_shape = stitching_function(name, current_label, current_shape,
                                                             current_maxlabel=maxlabel, start_index_2=(0, 0, x), res_path = res_path)
        coordinate[axis] += step
        z, y, x = coordinate
        end = coordinate[axis] >= total_shape[axis]
    current_shape = (min(total_shape[0] - start_index[0], step + 256), min(total_shape[1] - start_index[1], step + 256), total_shape[2])

    return current_shape, maxlabel

def stitching_column_by_rows(z = 0, column_index = 0, shapes_of_rows = [(1024, 1024, 1024)], maxlabel = 0, start_row_index = 0, end_row_index = 1,
                             step_z = 768, step_y = 768, step_x = 768, total_shape=(1024, 2304, 1792), res_path = None , overlap = 256):
    y = 0
    x = 0
    current_shape = (0, 0, 0)

    name = 'result_stitching_column_' + str(column_index) + '.h5'

    if os.path.exists(res_path + name) :
        current_shape = (min(total_shape[0] - z, step_z + overlap), total_shape[1], total_shape[2])
        f = h5py.File(res_path + name, 'a')
        bb = np.s_[: current_shape[0], :, :]
        maxlabel = np.max(f['data'][bb])
        return current_shape, maxlabel

    with h5py.File(res_path + name, mode='w') as f:
        f.create_dataset('data', shape=total_shape, compression='gzip', dtype='uint64')

    result_file_column= h5py.File(res_path + name, mode='a')
    result_dataset_column = result_file_column['data']
    for i in range(start_row_index, end_row_index):
        name_row = 'result_stitching_row_' + str(i) + '.h5'
        #name = 'data_row' + str(i)
        file_row = h5py.File(res_path + name_row, mode='r')
        current_label = file_row['data'][:shapes_of_rows[i][0], :  shapes_of_rows[i][1], :shapes_of_rows[i][2]]
        if current_shape == (0, 0, 0):
            current_shape = current_label.shape
            part = np.s_[ :  current_shape[0], : current_shape[1], : current_shape[2]]
            result_dataset_column.write_direct(current_label, np.s_[:, :, :], part)
        else:
            maxlabel, current_shape = stitching_function(name, current_label, current_shape,
                                                         current_maxlabel=maxlabel,
                                                         start_index_2=(0, y, 0), res_path = res_path)
        y += step_y
    return current_shape, maxlabel

def stitch_total(step_x=768, step_y=768, step_z=768, maxlabel=0, total_shape=(1024, 2304, 1792), n_workers=1, res_path = None , result_filename = None , overlap = 256):

    nz, ny, nx = total_shape
    row_index = 0
    shapes_of_rows = list()
    maxlabels = []
    y_positions = np.arange(0, ny, step_y)
    z_positions = np.arange(0, nz, step_z)

    n_rows = len(y_positions) * len(z_positions)
    row_indexes = np.arange(0, n_rows)

    variables = []
    i = 0
    for z in z_positions:
        for y in y_positions:
            variables.append((z, y, row_indexes[i]))
            i += 1

    with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        # submit the tasks
        tasks = [
            tpe.submit(stitching_row, start_index=(z, y, 0), row_count=row_index, step=step_x,
                        total_shape =total_shape, maxlabel=maxlabel, axis=2, res_path = res_path, overlap = overlap)
            for z, y, row_index in variables
        ]
        # get results
        results = [task.result() for task in tasks]
    print('row index', row_indexes[-1])
    print ('threads results after stitching rows ', results)
    for res in results:
        shapes_of_rows.append(res[0])
        maxlabels.append(res[1])
    maxlabel = maxlabels[0]

    shapes_of_columns = list()

    variables1 = []
    i = 0
    n_y_pos = len(y_positions)
    row_index = 0
    column_indexes = np.arange(0, len(z_positions))
    for z in z_positions:
        variables1.append((z, column_indexes[i], row_index, row_index + n_y_pos))
        i += 1
        row_index += n_y_pos

    with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        # submit the tasks
        tasks = [
            tpe.submit(stitching_column_by_rows, z = z, column_index = column_index, shapes_of_rows = shapes_of_rows,
                       maxlabel = maxlabel, start_row_index = start_row_index, end_row_index = end_row_index,
                       step_z = step_z, step_y = step_y, step_x = step_x, total_shape=total_shape, res_path = res_path, overlap = overlap)
            for z, column_index, start_row_index, end_row_index in variables1
        ]
        # get results
        results1 = [task.result() for task in tasks]

    maxlabels1 = []
    for res in results1:
        shapes_of_columns.append(res[0])
        maxlabels1.append(res[1])
    maxlabel = maxlabels1[0]

    z = 0
    y = 0
    x = 0
    current_shape = (0, 0, 0)
    result_file = h5py.File(res_path + result_filename, 'a')
    result_dataset = result_file['data']
    for i in range(len(column_indexes)):
        name = 'result_stitching_column_' + str(i) + '.h5'
        f = h5py.File(res_path + name)
        current_label = f['data'][: shapes_of_columns[i][0], : shapes_of_columns[i][1], :shapes_of_columns[i][2]]
        if current_shape == (0, 0, 0):
            current_shape = current_label.shape
            part = np.s_[: current_shape[0], : current_shape[1], : current_shape[2]]
            result_dataset.write_direct(current_label, np.s_[:, :, :], part)
        else:
            maxlabel, current_shape = stitching_function(result_filename, current_label, current_shape,
                                                         current_maxlabel=maxlabel,
                                                         start_index_2=(z, 0, 0), res_path =res_path)
        z += step_z
    return current_shape, maxlabel

def stitching_function(result_filename, new_region, current_shape, current_maxlabel=0, start_index_2=None, res_path = None ):

    new_region, max_label, mapping = vigra.analysis.relabelConsecutive(new_region.astype('uint64'),
                                                                       start_label=current_maxlabel + 1,
                                                                       keep_zeros=True)
    new_region = vigra.analysis.applyMapping(labels=new_region, mapping={current_maxlabel + 1: 0},
                                             allow_incomplete_mapping=True)

    current_maxlabel = max_label
    shape2 = new_region.shape
    index = np.s_[start_index_2[0]: start_index_2[0] + shape2[0],
            start_index_2[1]: start_index_2[1] + shape2[1],
            start_index_2[2]: start_index_2[2] + shape2[2]]

    resultfile = h5py.File(res_path + result_filename, mode='a')
    result_data = np.zeros(shape2)
    resultfile['data'].read_direct(result_data, index, np.s_[:, :, :])

    # get shapes
    shape1 = current_shape
    shape2 = new_region.shape

    shape = np.zeros(3).astype('int')
    for i in range(3):
        shape[i] = max(start_index_2[i] + shape2[i], shape1[i])

    shape = tuple(shape)

    # non-overlapping regions
    for i in range(3):
        if start_index_2[i] + shape2[i] > shape1[i]:
            start = [0, 0, 0]
            start[i] = shape1[i]
            part = np.s_[start[0]: start_index_2[0] + shape2[0], start[1]: start_index_2[1] + shape2[1],
                   start[2]: start_index_2[2] + shape2[2]]
            start[i] = shape1[i] - start_index_2[i]
            part1 = np.s_[start[0]: start_index_2[0] + shape2[0], start[1]: start_index_2[1] + shape2[1],
                    start[2]: start_index_2[2] + shape2[2]]
            # result_dataset.write_direct(new_region, source_sel = part1, dest_sel = part)
            result_data[part1] = np.copy(new_region[part1])

    # overlapping region
    start_in1 = np.array(start_index_2)
    end_in1 = np.array((min(start_in1[0] + shape2[0], shape1[0]),
                        min(start_in1[1] + shape2[1], shape1[1]),
                        min(start_in1[2] + shape2[2], shape1[2])))

    part_in1 = np.s_[start_in1[0]: end_in1[0], start_in1[1]: end_in1[1], start_in1[2]: end_in1[2]]

    start_in2 = np.array((0, 0, 0))
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

    if labels_overlap2.size == 1 and labels_overlap2 == [0.] and labels_overlap1.size == 1 and labels_overlap1 == [0.]:
        resultfile['data'].write_direct(result_data, np.s_[:, :, :], index)
        return (current_maxlabel + 1, shape)

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

    shape_overlap = result_data[part_in1].shape

    comparison = result_data[part_in1] * new_region[part_in2] != 0
    intersection_objects_new = comparison * new_region[part_in2]

    for i in labels_overlap2:
        if i != 0:
            match, counts = np.unique(result_data[part_in1][np.where(intersection_objects_new == i)], return_counts = True)
            n = match.size
            if n > 0:
                mapping_overlap[i] = match
                sizes_intercept[i] = counts

    iou = {}
    for l2 in mapping_overlap.keys():
        n = mapping_overlap[l2].size
        iou[l2] = np.zeros(n)
        for i in range(n):
            l1 = mapping_overlap[l2][i]
            iou[l2][i] = sizes_intercept[l2][i] / (sizes_overlap1[l1] + sizes_overlap2[l2] - sizes_intercept[l2][i])


    for l2 in labels_overlap2:
        comparison1 = (result_data[part_in1] == 0) * (new_region[part_in2] == l2)
        if l2 in mapping_overlap.keys():
            n = mapping_overlap[l2].size
            if n > 0:
                i = 0
                if n > 1:
                  i = np.argmax(iou[l2])
                if iou[l2][i] >= 0.8:
                    result_data[part_in1] = np.where(comparison1, mapping_overlap[l2][i], result_data[part_in1])
                else:
                    result_data[part_in1] = np.where(comparison1, l2, result_data[part_in1])
            else:
                result_data[part_in1] = np.where(comparison1, l2, result_data[part_in1])
        else:
            result_data[part_in1] = np.where(comparison1, l2, result_data[part_in1])


    for l2 in mapping_overlap.keys():
        n = mapping_overlap[l2].size
        i = 0
        if n > 1:
            i = np.argmax(iou[l2])
        if iou[l2][i] >= 0.8:
            result_data = np.where(result_data == l2, mapping_overlap[l2][i], result_data)

    resultfile['data'].write_direct(result_data, np.s_[:, :, :], index)
    return (current_maxlabel + 1, shape)

def run_stitching(filename_raw = None, dataset_raw = '/t00000/s00/0/cells', res_path = None, step = (768, 768, 768), n_workers = 1, overlap = 256):
    if not filename_raw :
        filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
    if not res_path:
        res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/'  # + s4a2_mc_z_y_x_shape.h5

    with h5py.File(filename_raw, 'r') as f:
        shape = f[dataset_raw].shape

    result_filename = 'result_stitching_file_final.h5'
    with h5py.File(res_path + result_filename, mode='w') as f:
        f.create_dataset('data', shape=shape, compression='gzip', dtype='uint64')

    if type(step) == tuple:
        step_z, step_y, step_x = step
    else :
        step_z = step
        step_y = step
        step_x = step

    current_shape, maxlabel = stitch_total(step_x=step_x, step_y=step_y, step_z=step_z, maxlabel=0, total_shape=shape, n_workers=n_workers,
                                           res_path = res_path, result_filename = result_filename, overlap = overlap)

run_stitching()