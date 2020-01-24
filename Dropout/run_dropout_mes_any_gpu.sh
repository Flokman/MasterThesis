#!/bin/bash
#SBATCH --time=00:45:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=MCDropoutMessidor2
#SBATCH --mem=32000

module load Python/3.6.4-fosscuda-2018a 
module load TensorFlow/1.12.0-fosscuda-2018a-Python-3.6.4 
module load scikit-learn/0.19.1-foss-2018a-Python-3.6.4 
module load Pillow/5.0.0-foss-2018a-Python-3.6.4

python dropout_mes.py