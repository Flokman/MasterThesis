#!/bin/bash
#SBATCH --time=00:15:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=create_dataset
#SBATCH --mem=8000

module load Python/3.6.4-foss-2018a
module load scikit-learn/0.19.1-foss-2018a-Python-3.6.4 
module load OpenCV/3.4.4-foss-2018a-Python-3.6.4

python create_dataset.py