#!/bin/bash
#SBATCH --time=00:30:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=jpg_to_png
#SBATCH --mem=8000

module load Python/3.6.4-foss-2018a
module load OpenCV/3.4.4-foss-2018a-Python-3.6.4
module load Pillow/5.0.0-foss-2018a-Python-3.6.4

python jpg_to_png.py