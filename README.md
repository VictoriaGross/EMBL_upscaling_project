# EMBL Upscaling Project

## Files 

Main files to use are: 
```
run_multicut.py
run_stiching.py
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
                  input_raw_files_train = '...', input_mem_files_train = '...', input_sv_files_train = '...', input_gt_files_train = '...',
                  
                  input_raw_file = '...', dataset_raw = 'data', input_mem_file = '...',

                  need_stitching_supervoxels = True, input_sv_path = '...', sv_pattern = '{}_{}_{}.h5',
                  step_sv = (256, 256, 256), overlap_sv = (256, 256, 256), smart_stitch_sv = False,
                   
                  n_threads = 1, 

                  beta = 0.5, result_pattern='mc_{}_{}_{}.h5', segm_cube_size = (1024, 1024, 1024),
                  segm_overlap=(256, 256, 256), segm_block_size=[256, 256, 256], 
                  need_stitching_results = False)
```
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
