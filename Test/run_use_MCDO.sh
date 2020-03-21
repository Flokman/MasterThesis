#!/bin/bash
#SBATCH --time=00:05:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=use_MCDO
#SBATCH --mem=32000

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4 
module load scikit-learn/0.21.3-foss-2019b-Python-3.7.4  
module load Pillow/6.2.1-GCCcore-8.3.0

python use_MCDO.py