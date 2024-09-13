# Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems


This repository contains a code that was used to produce results for the paper "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems" that was published at RecSys '24.



## Extract Model Embeddings

Code to extract pretrained embeddings is located in `extract_item_embeddings/`. 

For each model there are two files:

- `model.py` which reads files and calculates embeddings 
- `model.sh` which is a slurm job file that was used to submit the job to a cluster with a `sbatch model.sh`

Different models use different input sample rates that were precomputed with `ffmpeg`.

We store precomputed embeddings (not provided in this repository due to size) as `.npy` arrays. Therefore our internal index for tracks differs from original `track_id`. To get our index we sorted all `track_id` in the dataset and used index as a new id. The resulting mapping can be found in the `trackid_sorted.csv` 

## Train Test Split

- `0_get_plays_pqt.py` converts Music4All track_ids to our indexes
- `1_train_test_split.py` splits the log file into train/test/validation and compresses a raw log into a user-playcount format

## Model Definitions

- `bert4rec.py` contains model definition for BERT4Rec
- `bert4rec.yaml` contains default parameters for BERT4Rec, most of them are subject to be redefined for a training with `train_bert.sh`
- `model.py` defines `ShallowEmbeddingModel` which is referenced as Shallow Net in the paper
- `dataset.py` defines `InteractionDatasetItems` used to handle dataset.
- `knn.py` generates results for the KNN model

## Training

Training happens in `train.py` for Shallow Net and `train_bert.py` for BERT4Rec. We used slurm manager for submitting jobs and to we created `train.sh` and `train_bert.sh` to help with submitting many jobs with different parameters.

## Collecting Results

- `metrics.py` is an example on how the metrics were calculated for a run using [rs_metrics](https://github.com/Darel13712/rs_metrics)
- `table.py` takes calculated metrics for specified train runs and combines them into a single csv for all backend models
