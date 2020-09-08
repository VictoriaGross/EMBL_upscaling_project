import os
import pickle
import numpy as np
import elf
import elf.segmentation.workflows as elf_workflow
from elf.io import open_file
import h5py
import vigra
from run_stitching import *

def get_rf_model(
        output_folder = None,
        input_raw_files = None,
        input_mem_files = None,
        input_sv_files = None,
        input_gt_files = None):

    """
    :param output_folder: path to folder where to write a file with the model
    :param input_raw_files: list or just string describing path(s) to raw file(s) for training
    :param input_mem_files:  list or just string describing path(s) to membrane prediction file(s) for training
    :param input_sv_files:  list or just string describing path(s) to supervoxels file(s) for training
    :param input_gt_files:  list or just string describing path(s) to griund truth  file(s) for training
    :return: Random Forest model
    """
    if not output_folder :
        output_folder = '/g/schwab/Viktoriia/src/source/'
    rf_save_path = output_folder + 'rf.pkl'
    if os.path.exists(rf_save_path):
        with open(rf_save_path, 'rb') as f:
            rf = pickle.load(f)
            return rf

    if not input_raw_files:
        path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
        names = ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029', 's4a2_t033', 's4a2_t034', 's4a2_t035', 's4a2_t032']
        input_raw_files = [path_train + i + '/raw_crop_center256_256_256.h5' for i in names]
    if not input_mem_files:
        path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
        names = ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029', 's4a2_t033', 's4a2_t034', 's4a2_t035', 's4a2_t032']
        input_mem_files = [path_train + i + '/mem_crop_center256_256_256.h5' for i in names]
    if not input_sv_files:
        path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
        names = ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029', 's4a2_t033', 's4a2_t034', 's4a2_t035', 's4a2_t032']
        input_sv_files = [path_train + i + '/sv_crop_center256_256_256.h5' for i in names]
    if not input_gt_files:
        path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
        names = ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029', 's4a2_t033', 's4a2_t034', 's4a2_t035', 's4a2_t032']
        input_gt_files = [path_train + i + '/results/raw_MITO.h5' for i in names]

    if type(input_raw_files) != str:
        n = len(input_raw_files)
        assert len(input_mem_files) == n
        assert len(input_sv_files) == n
        assert len(input_gt_files) == n

        raw_train = []
        mem_train = []
        gt_train = []
        sv_train = []
        new_sv_labels_train = []

        for i in range(n):
            # raw data
            filename = input_raw_files[i]
            f = open_file(filename, 'r')
            raw_train.append(f['data'][:].astype(np.float32))

            # membrane prediction -- boundaries
            filename = input_mem_files[i]
            f = open_file(filename, 'r')
            mem_train.append(f['data'][:].astype(np.float32))

            # ground truth
            filename = input_gt_files[i]
            f = open_file(filename, 'r')
            gt_train.append(f['data'][:].astype(np.float32))

            # supervoxels
            filename = input_sv_files[i]
            f = open_file( filename, 'r')
            sv_train.append(f['data'][:].astype('uint64'))  # (np.float32)

            newlabels_train, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_train[-1])
            new_sv_labels_train.append(newlabels_train)
    else:
        #raw data
        f = open_file(input_raw_files, 'r')
        raw_train = f['data'][:].astype(np.float32)

        # membrane prediction -- boundaries
        f = open_file(input_mem_files, 'r')
        mem_train = f['data'][:].astype(np.float32)

        # ground truth
        f = open_file(input_gt_files, 'r')
        gt_train = f['data'][:].astype(np.float32)

        # supervoxels
        f = open_file(input_sv_files, 'r')
        sv_train = f['data'][:].astype('uint64')  # (np.float32)

        new_sv_labels_train, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_train)

    # do edge training
    rf = elf_workflow.edge_training(raw=raw_train, boundaries=mem_train, labels=gt_train,
                                        use_2dws=False, watershed=new_sv_labels_train)
    with open(rf_save_path, 'wb') as f:
        pickle.dump(rf, f)
    return rf


def concat_supervoxels(path_sv = None, stitched_sv_path = None, shape = None, step = (256, 256, 256), sv_pattern = '{}_{}_{}.h5', overlap = (256, 256, 256)):

    """
    Concatenation of center cubes
    :param path_sv: path to folder with input supervoxels files
    :param stitched_sv_path: path to resulting file with stitched supervoxels
    :param shape: shape of the raw dataset
    :param step: steps along each axis (tuple of size 3)
    :param sv_pattern: pattern like *{z}_{y}_{x}*.h5 to access blocks of supervoxels
    :param overlap: size of the overlap (tuple of size 3)
    :return:
    """
    with h5py.File(stitched_sv_path, mode='w') as f:
        f.create_dataset('data', shape=shape, compression='gzip', dtype='uint32')

    sv_file = h5py.File(stitched_sv_path, mode='a')
    sv_dataset = sv_file['data']
    max_label = 0

    for z in range(0, shape[0], step[0]):
        for y in range(0, shape[1], step[1]):
            for x in range(0, shape[2], step[2]):
                filename = sv_pattern.format(z, y, x)
                if os.path.exists(path_sv + filename):
                    f = open_file(path_sv + filename, 'r')
                    if len(list(f.keys())) != 0:
                        if shape[0] - z >= overlap[0] // 2 and shape[1] - y >= overlap[1] // 2 and shape[2] - x >= overlap[2] // 2:
                            start_sv = (overlap // 2 * int(z > 0), overlap // 2 * int(y > 0), overlap // 2 * int(x > 0))
                            end_sv = (min(shape[0] - overlap[0] // 2, shape[0] - z),
                                      min(shape[1] - overlap[1] // 2, shape[1] - y),
                                      min(shape[2] - overlap[2] // 2, shape[2] - x))
                            # part_sv = np.s_[start_sv[0]:end_sv[0], start_sv[1]:end_sv[1], start_sv[2]:end_sv[2]]
                            sv_current = f['data'][start_sv[0]:end_sv[0], start_sv[1]:end_sv[1],
                                         start_sv[2]:end_sv[2]].astype('uint32')
                            new_sv, new_max, mapping = vigra.analysis.relabelConsecutive(sv_current,
                                                                                         start_label=max_label + 1)
                            max_label = new_max
                            part = np.s_[z + start_sv[0]: z + end_sv[0], y + start_sv[1]: y + end_sv[1],
                                   x + start_sv[2]: x + end_sv[2]]
                            sv_dataset.write_direct(new_sv, np.s_[:, :, :], part)
    return


def stitch_supervoxels(input_sv_path = None, sv_pattern = '{}_{}_{}.h5',
                       input_raw_file = None, dataset_raw = '/t00000/s00/0/cells',
                       output_folder = None, smart_stitch = False,
                       step=(256, 256, 256), n_threads=1, overlap=256):
    """

    :param input_sv_path: path to folder with the input files with supervoxels
    :param sv_pattern: pattern of filenames like *{z}_{y}_{x}*.h5 to access blocks of supervoxels starting at z, y, x
    :param input_raw_file: path to the file with the initial raw dataset
    :param dataset_raw: internal dataset
    :param output_folder: path where to put results, stitching will be put into output_folder + 'sv_stitching/'
    :param smart_stitch: if True, stitching will be done in a way to oversee overlapping objects in the overlap zone; if False, center cubes will be concatenated
    :param step: steps along the 3 axis (tuple)
    :param n_threads: number of threads if doing smart stitching
    :param overlap: overlapping size (int)
    :return: full path to stitched file
    """
    if not input_sv_path:
        input_sv_path = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'
    if not input_raw_file:
        input_raw_file = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
    if not output_folder:
        output_folder_sv = '/g/schwab/Viktoriia/src/source/sv_stitching/'
    else:
        output_folder_sv = output_folder + 'sv_stitching/'
    result_filename = 'sv_total.h5'
    if smart_stitch:
        run_stitching(
            filename_raw=input_raw_file, dataset_raw=dataset_raw,
            output_folder = output_folder_sv,
            result_filename = result_filename,
            input_folder = input_sv_path,
            input_file_pattern = sv_pattern,
            step = step, n_threads=n_threads, overlap=overlap)
        return output_folder_sv + result_filename

    with open_file(input_raw_file, 'r') as f:
        shape = f[dataset_raw].shape
    concat_supervoxels(path_sv = input_sv_path, stitched_sv_path = output_folder_sv + result_filename,
                       shape = shape, step = step, sv_pattern = sv_pattern, overlap = overlap)
    return output_folder_sv + result_filename



def mc_segmentation(bb, mc_blocks = [256, 256, 256], filename_raw = None, dataset_raw = '/t00000/s00/0/cells',
                    filename_mem = None, filename_sv = None, rf = None,
                    n_threads = 4, beta = 0.5, error_file = None ):
    """

    :param bb: numpy slice that needs to be segmented
    :param mc_blocks: size of block for blockwise segmentation, list of size 3
    :param filename_raw: path to file with raw data
    :param dataset_raw: internal dataset
    :param filename_mem: path to file with membrane prediction
    :param filename_sv: path to file with supervoxels (stitched)
    :param rf: RandomForest model
    :param n_threads: number of threads (for multithreading)
    :param beta: beta value
    :param error_file: path to txt file where to write slice of the cube (bb) in case of runtimeerror
    :return: segmentation
    """
    f = open_file(filename_raw, 'r')
    data_raw = f[dataset_raw][bb].astype(np.float32)
    shape = data_raw.shape

    if np.min(data_raw) == np.max(data_raw):
        return np.zeros(shape)

    f = open_file(filename_mem, 'r')
    data_mem = f['data'][bb].astype(np.float32).reshape(data_raw.shape)
    assert data_mem.shape == shape

    f = open_file(filename_sv, 'r')
    data_sv = f['data'][bb].astype('uint32')
    assert data_sv.shape == shape

    data_sv, maxlabel, mapping = vigra.analysis.relabelConsecutive(data_sv)

    if np.min(data_sv) == np.max(data_sv):
        return np.zeros(shape)
    try:
        # run blockwise segmentation
        segmentation = elf_workflow.multicut_segmentation(raw=data_raw,
                                                          boundaries=data_mem,
                                                          rf=rf, use_2dws=False,
                                                          watershed=data_sv,
                                                          multicut_solver='blockwise-multicut',
                                                          solver_kwargs={'internal_solver': 'kernighan-lin',
                                                                         'block_shape': mc_blocks},
                                                          n_threads=n_threads, beta=beta)
        return segmentation
    except RuntimeError:
        with open(error_file, 'a') as f:
            f.write(str(bb) + '/n')
        return np.zeros(shape)

def run_segmentation_for_cubes(filename_raw = None, dataset_raw = '/t00000/s00/0/cells', filename_mem = None, filename_sv = None, rf = None,
                               n_threads = 1, beta = 0.5, output_folder = None, result_pattern = '{}_{}_{}.h5',
                               cube_size = (1024, 1024, 1024), overlap = (256, 256, 256), block_size = [256, 256, 256],
                               start_index = (0, 0, 0), end_index = None ) :

    step_z, step_y, step_x = cube_size[0] - overlap[0], cube_size[1] - overlap[1], cube_size[2] - overlap[2]
    with open_file(filename_raw, 'r') as f:
        shape = f[dataset_raw].shape
    nz, ny, nx = shape
    if not end_index:
        end_index = shape
    error_file = output_folder + 'errors_mc.txt'
    if not os.path.exists(error_file):
        with open(error_file, 'w') as f:
            f.write("Here will be indexes of cubes with errors in multicut. /n")
    for z in range(start_index[0], end_index[0], step_z):
        for y in range(start_index[1], end_index[1], step_y):
            for x in range(start_index[2], end_index[2], step_x):
                bb = np.s_[z: min(z + cube_size[0], nz), y: min(y + cube_size[1], ny), x: min(x + cube_size[2], nx)]
                filename_results = result_pattern.format(z, y, x)
                if not (os.path.exists(output_folder + filename_results)):
                    segmentation = mc_segmentation(bb, mc_blocks = block_size,
                                                   filename_raw = filename_raw, dataset_raw = dataset_raw,
                                                   filename_mem = filename_mem, filename_sv = filename_sv, rf = rf,
                                                   n_threads = n_threads, beta = beta, error_file = error_file)
                    f = open_file(output_folder + filename_results, 'w')
                    f.create_dataset('data', data=segmentation, compression="gzip")

def run_whole_process(output_folder = None,
                      input_raw_files_train = None, input_mem_files_train = None, input_sv_files_train = None, input_gt_files_train = None,
                      input_raw_file = None, dataset_raw = '/t00000/s00/0/cells', input_mem_file = None,
                      need_stitching_supervoxels = True, input_sv_path = None, sv_pattern = '{}_{}_{}.h5',
                      step_sv = (256, 256, 256), overlap_sv = (256, 256, 256), smart_stitch_sv = False,
                      sv_path = None,
                      n_threads = 1, beta = 0.5, result_pattern='mc_{}_{}_{}.h5', segm_cube_size = (1024, 1024, 1024),
                      segm_overlap=(256, 256, 256), segm_block_size=[256, 256, 256], segm_start_index = (0,0,0), segm_end_index = None,
                      need_stitching_results = False, result_filename_stitched = 'result_file_stitching_final.h5'):
    """

    :param output_folder: path to folder where to put all the intermediate files and the results
    :param input_raw_files_train: raw file or files for training (either a string or a list)
    :param input_mem_files_train: membrane prediction for the corresponding raw files for training
    :param input_sv_files_train: supervoxels for the corresponding files for training
    :param input_gt_files_train: ground truth for the corresponding files for training
    :param input_raw_file: path to the raw dataset that needs segmentation
    :param dataset_raw: internal dataset
    :param input_mem_file: membrane prediction
    :param need_stitching_supervoxels: True if cubes of supervoxels need to be stitched to get one file covering the whole dataset.
            if False, provide path to the already stitched file in sv_path
    :param input_sv_path: if need to stitch supervoxels, path to folder with blocks of supervoxels
    :param sv_pattern: if need to stitch supervoxels, pattern of supervoxel file names ({z}_{y}_{x}.h5)
    :param step_sv: steps along each axis used for generation of supervoxels
    :param overlap_sv: size of the overlap along each axis used for generation of supervoxels
    :param smart_stitch_sv: if True, stitching will be done in a way to oversee overlapping objects in the overlap zone; if False, center cubes will be concatenated
    :param sv_path: path to stitched supervoxels file
    :param n_threads: number of threads for multithreading
    :param beta: value of beta parameter for multicut segmentation
    :param result_pattern: pattern for writing results (mc_{z}_{y}_{x}.h5)
    :param segm_cube_size: size of cubes to be used for segmentation (1024, 1024, 1024)
    :param segm_overlap: size of overlap to leave along each axis while calculating segmentation
    :param segm_block_size: block size for blockwise multicut (256, 256, 256), must not be greater than cube size.
    :param segm_start_index: where to start segmentation, by default at (0,0,0)
    :param segm_end_index: where to end segmentation, by default at the end of dataset
    :param need_stitching_results: if True results will be stitched
    :param result_filename_stitched: filename for the stitched file, by default 'result_file_stitching_final.h5'
    :return:

    First we train random forest model. Then if needed stitch supervoxels. Then we run multicut segmentation. Then if needed stitch the results.

    There are two possibilities to stitch supervoxels -- concatenate centers or stitch with regards to overlapping region. Last one takes a long time and needs a lot of memmory.

    Multicut is done by cubes, with given overlap by the use of blockwise segmentation with given size of blocks.

    Stitching results can also be done here. But this would mean additional time and memory.
    """

    if not output_folder:
        output_folder = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/'

    #train RF
    rf = get_rf_model( output_folder=output_folder, input_raw_files=input_raw_files_train, input_mem_files=input_mem_files_train,
                       input_sv_files=input_sv_files_train, input_gt_files=input_gt_files_train)
    if not input_raw_file:
        input_raw_file = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
    if not input_mem_file:
        input_mem_file = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'

    #stitch supervoxels if needed
    if need_stitching_supervoxels:
        if not input_sv_path:
            input_sv_path = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'
        sv_path = stitch_supervoxels(input_sv_path=input_sv_path, sv_pattern=sv_pattern,
                           input_raw_file=input_raw_file, dataset_raw=dataset_raw,
                           output_folder=output_folder, smart_stitch=smart_stitch_sv,
                           step=step_sv, n_threads=n_threads, overlap=overlap_sv)
    #run_segmentation
    run_segmentation_for_cubes(filename_raw=input_raw_file, dataset_raw=dataset_raw, filename_mem=input_mem_file,
                               filename_sv=sv_path, rf=rf,
                               n_threads=n_threads, beta=beta, output_folder=output_folder, result_pattern=result_pattern,
                               cube_size=segm_cube_size, overlap=segm_overlap, block_size=segm_block_size,
                               start_index=segm_start_index, end_index=segm_end_index)

    if need_stitching_results:
        #stitch the results
        step = (segm_cube_size[0] - segm_overlap[0], segm_cube_size[1] - segm_overlap[1], segm_cube_size[2] - segm_overlap[2])
        run_stitching(
            filename_raw=input_raw_file, dataset_raw=dataset_raw,
            output_folder=output_folder,
            result_filename=result_filename_stitched,
            input_folder=output_folder,
            input_file_pattern=result_pattern,
            step=step, n_threads=n_threads, overlap=segm_overlap)
    return