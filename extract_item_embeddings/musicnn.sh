#!/bin/bash
#SBATCH --job-name="musicnn-emb"
#SBATCH --time=20:00:00
#SBATCH --ntasks=5
#SBATCH --mem=40G

module load any/python/3.8.3-conda
conda activate .musicnn_conda

python ./musicnn.py

