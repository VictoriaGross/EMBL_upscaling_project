#!/bin/bash
#SBATCH -A Schwab
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem 200000M
#SBATCH -t 5-00:00
#SBATCH -o multicut_mito_1.%N.%j.out
#SBATCH -e multicut_mito_1.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=viktoriia.gross@embl.de

source /g/schwab/Viktoriia/anaconda3/bin/activate /g/schwab/Viktoriia/envs/elf-env

which python

python ./mito_fix_segmentation.py

