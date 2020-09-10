from run_multicut import *

output_folder = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/updated_new_gt/'

path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
names = ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029', 's4a2_t033', 's4a2_t034', 's4a2_t035', 's4a2_t032']

input_raw_files_train  = [path_train + i + '/raw_crop_center256_256_256.h5' for i in names]
input_mem_files_train = [path_train + i + '/mem_crop_center256_256_256.h5' for i in names]
input_sv_files_train = [path_train + i + '/sv_crop_center256_256_256.h5' for i in names]
input_gt_files_train = [path_train + i + '/results/raw_MITO.h5' for i in names]

input_raw_file = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
input_mem_file = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'

#sv_path  = '/g/schwab/Viktoriia/src/source/s4a2_sv.h5'
input_sv_path = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'

run_whole_process(output_folder = None,
                  input_raw_files_train = input_raw_files_train,
                  input_mem_files_train = input_mem_files_train,
                  input_sv_files_train = input_sv_files_train,
                  input_gt_files_train = input_gt_files_train,

                  input_raw_file = input_raw_file, dataset_raw = '/t00000/s00/0/cells',
                  input_mem_file = input_mem_file,
                  need_stitching_supervoxels = True, input_sv_path = input_sv_path, sv_pattern = '{}_{}_{}.h5',
                  step_sv = (256, 256, 256), overlap_sv = (256, 256, 256), smart_stitch_sv = False,
                  n_threads = 4, beta = 0.6, result_pattern='mc_{}_{}_{}.h5', segm_cube_size = (1024, 1024, 1024),
                  segm_overlap=(256, 256, 256), segm_block_size=[512, 512, 512], segm_start_index = (0,0,0), segm_end_index = None,
                  need_stitching_results = False, result_filename_stitched = 'result_file_stitching_final.h5')