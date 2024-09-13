#!/bin/bash
#SBATCH --job-name="jb-emb"
#SBATCH --time=120:00:00
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load libsndfile
module load ffmpeg
source ~/.e/bin/activate
python ./jb.py

