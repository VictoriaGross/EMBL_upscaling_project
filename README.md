# EMBL Upscaling Project

## Files 

Main files to use are: 
```
run_multicut.py
run_mc_s4a2_mito_new_gt.py
run_stiching.py
sv_stitching.py
```

There is an html documentation for these two source files: 
```
_build/html/index.html
```

## Usage 
To run code you would need following packages installed: 
```
elf : https://github.com/constantinpape/elf
vigra : normally installed with elf 
numpy
h5py 
```
### Multicut 
Example: 
```
from run_multicut import * 

run_whole_process(output_folder = '...',
                  input_raw_files_train = '...', input_mem_files_train = '...', input_sv_files_train = '...', 
                  input_gt_files_train = '...',
                  
                  input_raw_file = '...', dataset_raw = 'data', input_mem_file = '...',

                  need_stitching_supervoxels = True, input_sv_path = '...', sv_pattern = '{}_{}_{}.h5',
                  step_sv = (256, 256, 256), overlap_sv = (256, 256, 256), smart_stitch_sv = False,
                   
                  n_threads = 1, 

                  beta = 0.5, result_pattern='mc_{}_{}_{}.h5', segm_cube_size = (1024, 1024, 1024),
                  segm_overlap=(256, 256, 256), segm_block_size=[256, 256, 256], 
                  need_stitching_results = False)
```
or use example in 'run_mc_s4a2_mito_new_gt.py'. 

Key idea:  
1. train random forest model. 
1. if needed stitch supervoxels (`need_stitching_supervoxels == True`): concatenate centers (`smart_stitch_sv == False`) or stitch with regards to overlapping region (`smart_stitch_sv == True`). Last one takes a long time and needs a lot of memmory.
1. run multicut segmentation by cubes of `segm_cube_size` size with `segm_overlap` overlap between each consecutive pair by the use of blockwise segmentation with blocks of `segm_block_size`. It can be done for a selected region of the dataset. In this case set up `segm_start_index` and `segm_end_index`
1. if needed stitch the results (`need_stitching_results == True`). 

Parameters | Description 
-----------|------------
output_folder | path to folder where to put all the intermediate files and the results
input_raw_files_train | raw file or files for training (either a string or a list)
input_mem_files_train | membrane prediction for the corresponding raw files for training
input_sv_files_train | supervoxels for the corresponding files for training
input_gt_files_train | ground truth for the corresponding files for training
input_raw_file | path to the raw dataset that needs segmentation
dataset_raw | internal dataset
input_mem_file | membrane prediction
need_stitching_supervoxels | True if cubes of supervoxels need to be stitched to get one file covering the whole dataset. If False, provide path to the already stitched file in sv_path
input_sv_path | if need to stitch supervoxels, path to folder with blocks of supervoxels
sv_pattern | if need to stitch supervoxels, pattern of supervoxel file names ({z}_{y}_{x}.h5)
step_sv | steps along each axis used for generation of supervoxels
overlap_sv | size of the overlap along each axis used for generation of supervoxels
smart_stitch_sv | if True, stitching will be done in a way to oversee overlapping objects in the overlap zone; if False, center cubes will be concatenated
sv_path | path to stitched supervoxels file if need_stitching_supervoxels is False 
n_threads | number of threads for multithreading
beta | value of beta parameter for multicut segmentation
result_pattern | pattern for writing results (mc_{z}_{y}_{x}.h5)
segm_cube_size | size of cubes to be used for segmentation (1024, 1024, 1024)
segm_overlap | size of overlap to leave along each axis while calculating segmentation
segm_block_size | block size for blockwise multicut (256, 256, 256), must not be greater than cube size.
segm_start_index | where to start segmentation, by default at (0,0,0)
segm_end_index | where to end segmentation, by default at the end of dataset
need_stitching_results | if True results will be stitched
result_filename_stitched | filename for the stitched file, by default 'result_file_stitching_final.h5'

Full list of parameters with explanations can be found in the code or html documentation.


### Stitching 
Example: 
```
from run_stitching import *

input_path = '...'
filename_raw = '...'
pattern = '{}_{}_{}.h5'
res_path = '...'

run_stitching(
    filename_raw=filename_raw, dataset_raw='data',
    output_folder = res_path,
    input_folder = input_path,
    input_file_pattern = pattern,
    step=(256, 256, 256), n_threads=4, overlap=256)
```
or use example in `sv_stitching.py`

Parameters | Description 
-----------|------------
filename_raw | full path to inital dataset
dataset_raw | internal dataset in the h5 file with the raw data
res_path | output folder
input_folder | input folder (folder with the results of segmentation that needs to be stitched)
input_file_pattern | pattern in a form *{z}_{y}_{x}*.h5 where z, y, x are the coordinates of the top left conner of a segmentation cube
step | steps along z, y, x axis between segmentation cubes; can be either tuple of 3 numbers or one number if it is the same for all the 3 axis
n_threads | number of threads for multi-threading for rows and columns
overlap | overlap between 2 cubes

Full list of parameters with explanations can be found in the code or html documentation.      