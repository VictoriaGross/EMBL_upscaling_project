import os
import numpy as np
import h5py
import vigra
from concurrent.futures import ThreadPoolExecutor
import sys

maxlabel = 0


def stitching_row(start_index=(0, 0, 0), row_count=0, step=512, total_shape=(1024, 2304, 1792), maxlabel=maxlabel,
                  axis=2, res_path=None, overlap=256,
                  input_folder=None, input_file_pattern=None):
    """
    This function stitches the input cubes into rows along x axis.

    :param start_index: where starts the first cube
    :param row_count: number of current row, needed for result file and for further stitching into columns
    :param step: step between consecutive segmentation cubes along x axis
    :param total_shape: shape of the whole dataset
    :param res_path: output folder
    :param overlap: overlap between consecutive segmentation cubes
    :param input_file_pattern: pattern to get input files
    :return: new maxlabel and shape of the row

    Results are writen to output folder under 'result_stitching_row_{row_count}.h5'
    """

    name = 'result_stitching_row_' + str(row_count) + '.h5'

    #if you restart because of some reason this part will save time and access the existing files. If the exising file is empty, it will be recalulated.
    # if os.path.exists(res_path + name):
    #     current_shape = (
    #     min(total_shape[0] - start_index[0], step + overlap), min(total_shape[1] - start_index[1], step + overlap),
    #     total_shape[2])
    #     f = h5py.File(res_path + name, 'a')
    #     bb = np.s_[:current_shape[0], : current_shape[1], : current_shape[2]]
    #     maxlabel = np.max(f['data'][bb])
    #     if maxlabel != 0:
    #       return current_shape, maxlabel

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

    """
    This part consecutively stitches input files. 
    If it is the first one opened, it is just written directly to the result file. If not, stitching function is run to add new opened region to the result file. 
    If the input file is empty, it is considered equal to zeros, so just the current shape of the resulting dataset is updated.  
    """
    while not end:
        filename_results = input_file_pattern.format(z, y, x)
        if os.path.exists(input_folder + filename_results):
            f = h5py.File(input_folder + filename_results, 'r')
            if len(list(f.keys())) != 0:
                current_label = f['data'][:, :, :]

                if current_shape == (0, 0, 0):
                    current_shape = current_label.shape
                    part = np.s_[:    current_shape[0], :  current_shape[1], : current_shape[2]]
                    current_label, new_maxlabel, mapping = vigra.analysis.relabelConsecutive(current_label.astype('uint64'),
                                                                                             start_label=maxlabel + 1,
                                                                                             keep_zeros=True)
                    current_label = vigra.analysis.applyMapping(current_label, mapping={maxlabel + 1: 0},
                                                                allow_incomplete_mapping=True)
                    maxlabel = new_maxlabel
                    result_dataset_row.write_direct(current_label, np.s_[:, :, :], part)
                else:
                    maxlabel, current_shape = stitching_function(name, current_label, current_shape,
                                                                 current_maxlabel=maxlabel, start_index_2=(0, 0, x),
                                                                 res_path=res_path)
            else:
                size = step + overlap
                if current_shape == (0, 0, 0):
                    current_shape = (size, size, size)
                else:
                    shape = list(current_shape)
                    shape[2] = min (total_shape[2], shape[2] + step)
                    current_shape = tuple(shape)
        coordinate[axis] += step
        z, y, x = coordinate
        end = coordinate[axis] >= total_shape[axis]
    # current_shape = (
    #     min(total_shape[0] - start_index[0], step + 256),
    #     min(total_shape[1] - start_index[1], step + 256), total_shape[2])

    return current_shape, maxlabel


def stitching_column_by_rows(z=0, column_index=0, shapes_of_rows=[(1024, 1024, 1024)], maxlabel=0, start_row_index=0,
                             end_row_index=1,
                             step_z=768, step_y=768, step_x=768, total_shape=(1024, 2304, 1792), res_path=None,
                             overlap=256):
    """
    This function stitches rows into columns along y axis.

    :param z: starting z for this column
    :param column_index: number of current column
    :param shapes_of_rows: shapes of rows to be stitched, should be a list of 2 tuples of size 3
    :param maxlabel: starting label for this column
    :param start_row_index: index of the first row in the selection of rws to be stitched together
    :param end_row_index: index of the last row in the selection of rws to be stitched together (not included in result)
    :param step_z: step along z axis
    :param step_y: step along y axis
    :param step_x: step along x axis
    :param total_shape: total shape of the inital dataset
    :param res_path: folder for the output files
    :param overlap: overlap between consecutive input cubes of segmentation
    :return: new maxlabel and shape of the row

    Results are writen to output folder under 'result_stitching_column_{column_index}.h5'
    """
    y = 0
    x = 0
    current_shape = (0, 0, 0)

    name = 'result_stitching_column_' + str(column_index) + '.h5'

    # if you restart because of some reason this part will save time and access the existing files. If the exising file is empty, it will be recalulated.
    # if os.path.exists(res_path + name):
    #     current_shape = (min(total_shape[0] - z, step_z + overlap), total_shape[1], total_shape[2])
    #     f = h5py.File(res_path + name, 'a')
    #     bb = np.s_[: current_shape[0], :, :]
    #     maxlabel = np.max(f['data'][bb])
    #     if maxlabel != 0:
    #       return current_shape, maxlabel

    with h5py.File(res_path + name, mode='w') as f:
        f.create_dataset('data', shape=total_shape, compression='gzip', dtype='uint64')

    result_file_column = h5py.File(res_path + name, mode='a')
    result_dataset_column = result_file_column['data']
    """
    This part consecutively stitches row files . 
    If it is the first one opened, it is just written directly to the result file. If not, stitching function is run to add new opened region to the result file. 
    """
    for i in range(start_row_index, end_row_index):
        name_row = 'result_stitching_row_' + str(i) + '.h5'
        file_row = h5py.File(res_path + name_row, mode='r')
        current_label = file_row['data'][:shapes_of_rows[i][0], :  shapes_of_rows[i][1], :shapes_of_rows[i][2]]
        if current_shape == (0, 0, 0):
            current_shape = current_label.shape
            part = np.s_[:  current_shape[0], : current_shape[1], : current_shape[2]]
            result_dataset_column.write_direct(current_label, np.s_[:, :, :], part)
        else:
            maxlabel, current_shape = stitching_function(name, current_label, current_shape,
                                                         current_maxlabel=maxlabel,
                                                         start_index_2=(0, y, 0), res_path=res_path)
        y += step_y
    return current_shape, maxlabel


def stitch_total(step_x=768, step_y=768, step_z=768, maxlabel=0, total_shape=(1024, 2304, 1792), n_workers=1,
                 res_path=None, result_filename=None, overlap=256,
                 input_folder=None, input_file_pattern=None):
    """
     This function first stitches cubes into rows, then rows into columns and then the final step --- all together.

    :param step_x, step_y, step_z: steps used for segmentation calculation along the 3 axis
    :param maxlabel: all new labels will be greater or equal to this one
    :param total_shape: shape of the whole dataset
    :param n_workers: number of threads
    :param res_path: folder for writing results
    :param result_filename: name of the final file with results
    :param overlap: overlap between cubes used for segmentation calculations
    :param input_folder: folder with segmentation results to be stitched
    :param input_file_pattern: pattern '*{}_{}_{}*.h5' where {} correspond to z, y, x
    :return: new maxlabel and shape of the row


    """
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
    #stitching into rows with the use of multithreading
    with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        # submit the tasks
        tasks = [
            tpe.submit(stitching_row, start_index=(z, y, 0), row_count=row_index, step=step_x,
                       total_shape=total_shape, maxlabel=maxlabel, axis=2, res_path=res_path, overlap=overlap,
                       input_folder=input_folder, input_file_pattern=input_file_pattern)
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
    #stitching rows into columns by multithreading
    with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        # submit the tasks
        tasks = [
            tpe.submit(stitching_column_by_rows, z=z, column_index=column_index, shapes_of_rows=shapes_of_rows,
                       maxlabel=maxlabel, start_row_index=start_row_index, end_row_index=end_row_index,
                       step_z=step_z, step_y=step_y, step_x=step_x, total_shape=total_shape, res_path=res_path,
                       overlap=overlap)
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
    #stitch everything up
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
                                                         start_index_2=(z, 0, 0), res_path=res_path)
        z += step_z
    return current_shape, maxlabel


def stitching_function(result_filename, new_region, current_shape, current_maxlabel=0, start_index_2=None,
                       res_path=None):

    """
    This function stitches existing region and new one taking in consideration the overlapping region.

    :param result_filename: where to write results and where to add new region
    :param new_region: new region to be added to the result file
    :param current_shape: shape of the current region in the result file
    :param current_maxlabel: used to relabel new region without intersection of the labels
    :param start_index_2: where new region starts
    :param res_path: folder fo the output
    :return: new maxlabel and new current shape
    """

    """
    First we relabel new region to consecutive labels, that are greater than labels in the existing region. 
    """
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

    """From the file with the results we read only the part that corresponds to the new region, to get information about overlapping region, and to have a part that will cover results for the non-overlapping region."""
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
    """
    Then we deal with non-overlapping region of the new one. It is copied to the result dataset. 
    """
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
    """
    Here we start with the overlapping zone. 
    """
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

    """ We get the unique lqbels in this area from both existing and new regions. """
    labels_overlap1 = np.unique(result_data[part_in1])
    labels_overlap2 = np.unique(new_region[part_in2])

    if labels_overlap2.size == 1 and labels_overlap2 == [0.] and labels_overlap1.size == 1 and labels_overlap1 == [0.]:
        resultfile['data'].write_direct(result_data, np.s_[:, :, :], index)
        return (current_maxlabel + 1, shape)

    """
    We create a mapping that maps labels from new region to the labels in existing one if they intersect. 
    We also stock sizes of corresponding objects in the respective areas. 
    And we also stock the sizes of intersection of labels. 
    """
    mapping_overlap = {}
    sizes_overlap1 = {}
    sizes_overlap2 = {}
    sizes_intercept = {}

    #sizes of objects in existing area
    for i in labels_overlap1:
        if i != 0:
            sizes_overlap1[i] = np.count_nonzero(result_data[part_in1] == i)

    #sizes of objects in new region
    for i in labels_overlap2:
        if i != 0:
            sizes_overlap2[i] = np.count_nonzero(new_region[part_in2] == i)

    shape_overlap = result_data[part_in1].shape

    comparison = result_data[part_in1] * new_region[part_in2] != 0
    intersection_objects_new = comparison * new_region[part_in2] #region where both labels from the existing and new regions are not zeros, and value is equal to the one from new region

    #create mapping and measure intersection
    for i in labels_overlap2:
        if i != 0:
            match, counts = np.unique(result_data[part_in1][np.where(intersection_objects_new == i)],
                                      return_counts=True)
            n = match.size
            if n > 0:
                mapping_overlap[i] = match
                sizes_intercept[i] = counts

    """ We calculate iou for each item in mapping. """
    iou = {}
    for l2 in mapping_overlap.keys():
        n = mapping_overlap[l2].size
        iou[l2] = np.zeros(n)
        for i in range(n):
            l1 = mapping_overlap[l2][i]
            iou[l2][i] = sizes_intercept[l2][i] / (sizes_overlap1[l1] + sizes_overlap2[l2] - sizes_intercept[l2][i])

    """
    For each label from new region in intersection: 
        - if it is in mapping and has corresponding labels in the existing area, it is relabelled to the label from exisiting region if IOU is greater than 0.8. 
        - if all the corresponding labels in the existing area correspond to IOU less than 0.8, this label from new region is conserved in the area avoiding the intersection with other object from existing region. 
        - if there was no intersection with any other labels from existing region, we preserve the object. 
    """
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

    """ 
    For each label from new region in intersection, if there was a match with a label from the existing region that had IOU greater than 0.8, this label is relabeled in the non-overlapping part.   
    """
    for l2 in mapping_overlap.keys():
        n = mapping_overlap[l2].size
        i = 0
        if n > 1:
            i = np.argmax(iou[l2])
        if iou[l2][i] >= 0.8:
            result_data = np.where(result_data == l2, mapping_overlap[l2][i], result_data)

    """Result is written into the output file directly. """
    resultfile['data'].write_direct(result_data, np.s_[:, :, :], index)
    return (current_maxlabel + 1, shape)


def run_stitching(filename_raw=None, dataset_raw='/t00000/s00/0/cells', output_folder =None, input_folder=None,
                  input_file_pattern=None, step=(768, 768, 768), n_threads=1, overlap=256):
    """
    :param filename_raw: full path to inital dataset
    :param dataset_raw: internal dataset in the h5 file with the raw data
    :param res_path: output folder
    :param input_folder: input folder (folder with the results of segmentation that needs to be stitched)
    :param input_file_pattern: pattern in a form *{z}_{y}_{x}*.h5 where z, y, x are the coordinates of the top left conner of a segmentation cube
    :param step: steps along z, y, x axis between segmentation cubes; can be either tuple of 3 numbers or one number if it is the same for all the 3 axis
    :param n_threads: number of threads for multi-threading for rows and columns
    :param overlap: overlap between 2 cubes
    :return:
    """
    if not filename_raw:
        filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
    if not output_folder:
        output_folder = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/' 
    if not input_folder:
        input_folder = output_folder
    if not input_file_pattern:
        input_file_pattern = 's4a2_mc_{}_{}_{}_1024_1024_1024.h5'
    with h5py.File(filename_raw, 'r') as f:
        shape = f[dataset_raw].shape

    #create file with results
    result_filename = 'result_stitching_file_final.h5'
    with h5py.File(output_folder + result_filename, mode='w') as f:
        f.create_dataset('data', shape=shape, compression='gzip', dtype='uint64')

    if type(step) == tuple:
        step_z, step_y, step_x = step
    else:
        step_z = step
        step_y = step
        step_x = step


    #run the stitching function
    current_shape, maxlabel = stitch_total(step_x=step_x, step_y=step_y, step_z=step_z, maxlabel=0, total_shape=shape,
                                           n_workers=n_threads,
                                           res_path=output_folder, result_filename=result_filename, overlap=overlap,
                                           input_folder=input_folder, input_file_pattern=input_file_pattern)
    return


