#!/bin/bash
#SBATCH --time=30:00:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=MCBNMessidor2
#SBATCH --mem=64000

module load Python/3.6.4-fosscuda-2018a 
module load TensorFlow/1.12.0-fosscuda-2018a-Python-3.6.4 
module load scikit-learn/0.19.1-foss-2018a-Python-3.6.4 
module load Pillow/5.0.0-foss-2018a-Python-3.6.4

python batch_normalization_mes.py