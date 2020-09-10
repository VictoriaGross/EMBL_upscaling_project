#!/bin/bash
#SBATCH -A Schwab
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 370G
#SBATCH -t 15-00:00
#SBATCH -o mc+stitching.%N.%j.out
#SBATCH -e mc+stitching.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=viktoriia.gross@embl.de

source /g/schwab/Viktoriia/anaconda3/bin/activate /g/schwab/Viktoriia/envs/elf-env

which python

python /g/schwab/viktoriia/src/source/run_mc_s4a2_mito_new_gt.py

