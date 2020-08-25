import os
import numpy as np
import elf
from elf.io import open_file
import h5py
import vigra
from concurrent.futures import ThreadPoolExecutor
import elf.segmentation.workflows as elf_workflow
from memory_profiler import  profile
import sys
import pickle

# filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'
#filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
filename_raw = '/scratch/vgross/amst_inv_clip_black_bg.h5'

# res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/'  # + s4a2_mc_z_y_x_shape.h5
#res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/updated_beta_0.6/'  # + s4a2_mc_z_y_x_shape.h5
#res_path = '/scratch/vgross/upscaling_results_dmv/'
res_path = '/scratch/vgross/upscaling_results_mito/updated_beta_0.6/'

with open_file(filename_raw, 'r') as f:
    # raw = f['/t00000/s00/0/cells'][:, :, :].astype(np.float32)
    shape = f['/t00000/s00/0/cells'].shape

print("Full  shape:", shape)

nz, ny, nx = shape

with h5py.File(res_path + 'result_stitching_file_final.h5', mode='w') as f:
    f.create_dataset('data', shape=shape, compression='gzip', dtype='uint64')
#
# result_filename = 'result_stitching_file_final.h5'
# result_file = h5py.File(res_path + result_filename, mode='a')
# result_dataset = result_file['data']

maxlabel = 0

#@profile
def stitching_row(start_index=(0, 0, 0), row_count=0, step=512, total_shape=(1024, 2304, 1792)): #, maxlabel=maxlabel, axis=2, use_stitched=False):
    name = 'result_stitching_row_' + str(row_count) + '.h5'
    if os.path.exists(res_path + name) :
        current_shape = (min(total_shape[0] - start_index[0], step + 256), min(total_shape[1] - start_index[1], step + 256), total_shape[2])
        #f = open_file(res_path + name, 'a')
        # bb = np.s_[start_index[0] : start_index[0] + current_shape[0], start_index[1] : start_index[1] + current_shape[1],
        #                             start_index[2] : start_index[2] + current_shape[2]]
        #bb = np.s_[:current_shape[0], : current_shape[1], : current_shape[2]]
        #maxlabel = np.max(f['data'][bb])
        print("row ", start_index, current_shape)#, maxlabel)
        return current_shape#, maxlabel

    with h5py.File(res_path + name, mode='w') as f:
        f.create_dataset('data', shape=total_shape, compression='gzip', dtype='uint64')

    result_file_row = h5py.File(res_path + name, mode='a')
    result_dataset_row = result_file_row['data']

    print('row', row_count, start_index, total_shape)
    z = start_index[0]
    y = start_index[1]
    x = start_index[2]
    coordinate = list(start_index)
    end = False
    axis = 2
    nz, ny, nx = total_shape
    size_cube = step + 256
    current_shape = (0, 0, 0)
    maxlabel = 0
    while not end:
        filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' + '1024_1024_1024.h5'  # str(size_cube) + '_' + str(size_cube) + '_' + str(size_cube) + '.h5'
        if os.path.exists(res_path + filename_results):
            f = open_file(res_path + filename_results, 'r')
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
                print('done', z, y, x)
            else:
                try:
                    maxlabel, current_shape = stitching_function(name, current_label, current_shape,
                                                             current_maxlabel=maxlabel, start_index_2=(0, 0, x))
                    print('done', z, y, x)
                except:
                    print('some problem at ', z, y, x)
                    print(sys.exc_info())
        coordinate[axis] += step
        z, y, x = coordinate
        end = coordinate[axis] >= total_shape[axis]
        print('coordinate ', coordinate)
    print('done row ', row_count)
    current_shape = (min(total_shape[0] - start_index[0], step + 256), min(total_shape[1] - start_index[1], step + 256), total_shape[2])

    return current_shape #, maxlabel


def stitching_column_by_rows(z = 0, column_index = 0, shapes_of_rows = [(1024, 1024, 1024)], start_row_index = 0, end_row_index = 1, step_z = 768, step_y = 768, step_x = 768, total_shape=(1024, 2304, 1792)):
    y = 0
    x = 0
    current_shape = (0, 0, 0)

    name = 'result_stitching_column_' + str(column_index) + '.h5'

    if os.path.exists(res_path + name) :
        current_shape = (min(total_shape[0] - z, step_z + 256), total_shape[1], total_shape[2])
        #f = open_file(res_path + name, 'a')
        #bb = np.s_[: current_shape[0], :, :]
        #maxlabel = np.max(f['data'][bb])
        print("column ", column_index, z, current_shape) #, maxlabel)
        return current_shape #, maxlabel

    with h5py.File(res_path + name, mode='w') as f:
        f.create_dataset('data', shape=shape, compression='gzip', dtype='uint64')

    result_file_column= h5py.File(res_path + name, mode='a')
    result_dataset_column = result_file_column['data']
    maxlabel = 0
    print('column ', column_index)
    for i in range(start_row_index, end_row_index):
        name_row = 'result_stitching_row_' + str(i) + '.h5'
        #name = 'data_row' + str(i)
        file_row = open_file(res_path + name_row, mode='r')
        current_label = file_row['data'][:shapes_of_rows[i][0], :  shapes_of_rows[i][1], :shapes_of_rows[i][2]]
        if current_shape == (0, 0, 0):
            current_shape = current_label.shape
            part = np.s_[ :  current_shape[0], : current_shape[1], : current_shape[2]]
            result_dataset_column.write_direct(current_label, np.s_[:, :, :], part)
            print('done ', z, y, x)
        else:
            try:
                maxlabel, current_shape = stitching_function(name, current_label, current_shape,
                                                         current_maxlabel=maxlabel,
                                                         start_index_2=(0, y, 0))
                print('done ', z, y, x)
            except:
                 print('some problem at ', z, y, x)
                 print(sys.exc_info())
        y += step_y
    print('done column ', column_index)
    return current_shape #, maxlabel

def stitching_two_columns (column1 = 0, column2 = 1, column_index = 7, shapes_of_columns = [(1024, 3144,3864), (1024, 3144,3864)], start_index1 = (0,0,0), start_index2 = (768, 0,0), total_shape = (4824, 3144, 3864)):
    name1 = 'result_stitching_column_' + str(column1) + '.h5'
    name2 = 'result_stitching_column_' + str(column2) + '.h5'
    name_result = 'result_stitching_column_' + str(column_index) + '.h5'

    with h5py.File(res_path + name_result, mode='w') as f:
        f.create_dataset('data', shape=shape, compression='gzip', dtype='uint64')

    result_file_column= h5py.File(res_path + name_result, mode='a')
    result_dataset_column = result_file_column['data']

    f = open_file(res_path + name1)
    label1 = f['data'][: shapes_of_columns[0][0], : shapes_of_columns[0][1], :shapes_of_columns[0][2]]
    current_shape = shapes_of_columns[0]
    part = np.s_[: current_shape[0], : current_shape[1], : current_shape[2]]
    result_dataset_column.write_direct(label1, np.s_[:, :, :], part)
    maxlabel = np.max(label1)

    f = open_file(res_path + name2)
    label2 = f['data'][: shapes_of_columns[1][0], : shapes_of_columns[1][1], :shapes_of_columns[1][2]]
    maxlabel, current_shape = stitching_function(name_result, label2, current_shape, current_maxlabel= maxlabel,
                                start_index_2 = (start_index2[0] - start_index1[0], start_index2[1] - start_index1[1], start_index2[2] - start_index1[2]))
    return current_shape, maxlabel

def stitch_total(step_x=768, step_y=768, step_z=768, maxlabel=0, total_shape=(1024, 2304, 1792), n_workers=1):


    shapes_of_rows = list()
    #maxlabels = []
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
                        total_shape =total_shape)#, maxlabel=maxlabel, axis=2)
            for z, y, row_index in variables
        ]
        # get results
        results = [task.result() for task in tasks]
    print('row index', row_indexes[-1])
    print ('threads results after stitching rows ', results)
    for res in results:
        shapes_of_rows.append(res[0])
        #maxlabels.append(res[1])
    #maxlabel = maxlabels[0]

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
                        start_row_index = start_row_index, end_row_index = end_row_index,
                        step_z = step_z, step_y = step_y, step_x = step_x, total_shape=total_shape)
            for z, column_index, start_row_index, end_row_index in variables1
        ]
        # get results
        results1 = [task.result() for task in tasks]

    print ('threads results after stitching columns ', results1)
    maxlabels1 = []
    for res in results1:
        shapes_of_columns.append(res[0])
        #maxlabels1.append(res[1])
    #maxlabel = maxlabels1[0]

    #current_shape = (0, 0, 0)

    variables = [] #column1 = 0, column2 = 1, column_index = 7, shapes_of_columns = [(1024, 3144,3864), (1024, 3144,3864)], start_index1 = (0,0,0), start_index2 = (768, 0,0)

    n = len(column_indexes)
    number_of_columns_inside = np.ones(n)
    current_z_position = z_positions
    max_index = n - 1
    current_indexes = column_indexes
    operations_row = []
    while n > 1:
        new_indexes = []
        new_numbers = []
        new_z = []
        k = 0
        for i in range(0, n - 1, 2):
            z1 = current_z_position[i]
            z2 = current_z_position[i + 1]
            shape1 = (int(min(total_shape[0] - z1, number_of_columns_inside[i] * step_z + 256)), total_shape[1], total_shape[2])
            shape2 = (int(min(total_shape[0] - z2, number_of_columns_inside[i + 1] * step_z + 256)), total_shape[1], total_shape[2])
            variables.append((current_indexes[i], current_indexes[i + 1], max_index + 1, list([shape1, shape2]),
                              (z1, 0, 0), (z2, 0, 0)))
            new_indexes.append(max_index + 1)
            new_numbers.append(number_of_columns_inside[i] + number_of_columns_inside[i + 1])
            new_z.append(z1)
            max_index += 1
            k += 1
        if n % 2 == 1:
            new_indexes.append(n - 1)
            new_numbers.append(number_of_columns_inside[-1])
            new_z.append(current_z_position[-1])
        current_indexes = new_indexes
        number_of_columns_inside = new_numbers
        current_z_position = new_z
        n = len(current_indexes)
        operations_row.append(k)
    ind = 0
    results = []
    for i in range(len(operations_row)):
        k = operations_row[i]
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            # submit the tasks
            tasks = [tpe.submit(stitching_two_columns, column1 = column1, column2 = column2, column_index = column_index,
                                shapes_of_columns = shapes_of_columns, start_index1 = start_index1, start_index2 = start_index2, total_shape=total_shape)
                for column1 , column2 , column_index, shapes_of_columns, start_index1, start_index2 in variables[ind:ind + k]
            ]
            # get results
            results = results + [task.result() for task in tasks]
    current_shape = results[-1][0]
    #maxlabel = results[-1][0]

    print('done total')
    return current_shape, current_indexes[0]

#@profile
def stitching_function(result_filename, new_region, current_shape, current_maxlabel=0, start_index_2=None):

    new_region, max_label, mapping = vigra.analysis.relabelConsecutive(new_region.astype('uint64'),
                                                                       start_label=current_maxlabel + 1,
                                                                       keep_zeros=True)
    new_region = vigra.analysis.applyMapping(labels=new_region, mapping={current_maxlabel + 1: 0},
                                             allow_incomplete_mapping=True)

    current_maxlabel = max_label
    # print('new max label', max_label)

    shape2 = new_region.shape

    #result_data = np.zeros(shape2)
    index = np.s_[start_index_2[0]: start_index_2[0] + shape2[0],
            start_index_2[1]: start_index_2[1] + shape2[1],
            start_index_2[2]: start_index_2[2] + shape2[2]]

    resultfile = h5py.File(res_path + result_filename, mode='a')
    result_data = np.zeros(shape2)
    resultfile['data'].read_direct(result_data, index, np.s_[:, :, :])
    #print('done reading', start_index_2)

    # get shapes
    shape1 = current_shape
    shape2 = new_region.shape

    shape = np.zeros(3).astype('int')
    for i in range(3):
        shape[i] = max(start_index_2[i] + shape2[i], shape1[i])

    shape = tuple(shape)


    # non-overlapping regions
    #print('start non-overlap')
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
            # print(part1)
            result_data[part1] = np.copy(new_region[part1])
    #print('done with non-overlapping', start_index_2)

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

    #print('labels_overlap_1', labels_overlap1, start_index_2)
    #print('labels overlap 2', labels_overlap2, start_index_2)

    if labels_overlap2.size == 1 and labels_overlap2 == [0.] and labels_overlap1.size == 1 and labels_overlap1 == [0.]:
        resultfile['data'].write_direct(result_data, np.s_[:, :, :], index)
        #print('no overlap region', start_index_2)
        return (current_maxlabel + 1, shape)

    # print ('part in 1', part_in1)
    # print ('part in 2', part_in2)
    # print ('part in overall ', part_new)

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

    # print('done with sizes of each object')

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

    #print('mapping ', mapping_overlap, start_index_2)

    iou = {}
    for l2 in mapping_overlap.keys():
        n = mapping_overlap[l2].size
        iou[l2] = np.zeros(n)
        for i in range(n):
            l1 = mapping_overlap[l2][i]
            iou[l2][i] = sizes_intercept[l2][i] / (sizes_overlap1[l1] + sizes_overlap2[l2] - sizes_intercept[l2][i])
    #print('done iou')


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

    #print('done overlap', start_index_2)

    for l2 in mapping_overlap.keys():
        n = mapping_overlap[l2].size
        i = 0
        if n > 1:
            i = np.argmax(iou[l2])
        if iou[l2][i] >= 0.8:
            result_data = np.where(result_data == l2, mapping_overlap[l2][i], result_data)

    resultfile['data'].write_direct(result_data, np.s_[:, :, :], index)
    #print ('done ', start_index_2)
    return (current_maxlabel + 1, shape)


current_shape,  index = stitch_total(step_x=768, step_y=768, step_z=768, maxlabel=0, total_shape=shape, n_workers=1)

