#!/bin/bash
#SBATCH --job-name="m2v-emb"
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=m2v.out

module load libsndfile
module load ffmpeg
source ~/.e/bin/activate
python ./m2v.py

