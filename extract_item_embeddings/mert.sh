#!/bin/bash
#SBATCH --job-name="mert-emb"
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=mert.out

module load libsndfile
module load ffmpeg
source ~/.e/bin/activate
python ./mert.py

