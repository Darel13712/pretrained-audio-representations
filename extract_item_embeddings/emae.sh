#!/bin/bash
#SBATCH --job-name="encodec-emb"
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load libsndfile
module load ffmpeg
source ~/.e/bin/activate
python ./emae.py

