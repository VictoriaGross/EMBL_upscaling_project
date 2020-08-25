#!/bin/bash
#SBATCH -A Schwab
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 370G
#SBATCH -t 5-00:00
#SBATCH -o stitching_mito.%N.%j.out
#SBATCH -e stitching_mito.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=viktoriia.gross@embl.de

source /g/schwab/Viktoriia/anaconda3/bin/activate /g/schwab/Viktoriia/envs/elf-env

which python

python /g/schwab/Viktoriia/src/source/stitching_mito.py

