#!/bin/bash
#SBATCH --time=00:10:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=MCDropoutMessidor2
#SBATCH --mem=32000

module load Python/3.6.4-foss-2018a
module load TensorFlow/1.12.0-fosscuda-2018a-Python-3.6.4 
module load scikit-learn/0.19.1-foss-2018a-Python-3.6.4 

python dropoutMes.py