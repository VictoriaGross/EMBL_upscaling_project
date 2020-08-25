import mobie

# root folder for the mobie project
mobie_root = '/g/schwab/Viktoriia/mobie-test/'
# name of the dataset to be added
dataset_name = 's4a2_mc'
# internal name for the initial data for this dataset
#data_name = 'raw_data'

# file path and key for the input data
# key can be an internal path for hdf5 or zarr/n5 containers
# or a file pattern for image stacks
# data_path = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
# data_key = '/t00000/s00/0/cells'
#
# # resolution of this initial data (in micrometer), chunks size and factors for down-scaling
resolution = (8, 8, 8)
chunks = (64, 128, 128)
scale_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
#
# # initialize a dataset for this project from some raw data
# mobie.initialize_dataset(data_path, data_key,
#                          mobie_root, dataset_name, data_name,
#                          resolution, chunks, scale_factors)
#
# segm_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/result_stitching_file_final.h5'
# segm_key = 'data'
# segm_name = 'mitochondria-stitched'
# mobie.add_segmentation(segm_path, segm_key, mobie_root, dataset_name, segm_name,
#                          resolution, scale_factors, chunks)

segm_path = '/g/schwab/hennies/project_corona/segmentation/upscale/20-04-23_S4_area2_Sam/seg_mito_scale1.0_2/run_200617_00_mito_full_dataset.h5'
segm_key = 'scale_1'
segm_name = 'mitochondria_julian_1'
mobie.add_segmentation(segm_path, segm_key, mobie_root, dataset_name, segm_name,
                         resolution, scale_factors, chunks)

segm_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_dmv/result_stitching_file_final.h5'
segm_key = 'data'
segm_name = 'dmv'
mobie.add_segmentation(segm_path, segm_key, mobie_root, dataset_name, segm_name,
                         resolution, scale_factors, chunks)