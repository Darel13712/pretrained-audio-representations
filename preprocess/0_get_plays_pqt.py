import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

input_emb_path = glob('embeddings/*.npy')
track_ids = pd.read_csv('embeddings/trackid_sorted.csv', index_col=0)['trackid']
df = pd.read_csv('../m4a/userid_trackid_timestamp.tsv', sep='\t', parse_dates=['timestamp'])


trackid_to_idx = pd.Series(data=track_ids.index, index=track_ids.values)
chunk_size = 1000000  # Adjust based on your system's memory
mapped_chunks = []

for chunk in tqdm(pd.read_csv('../m4a/userid_trackid_timestamp.tsv', sep='\t', parse_dates=['timestamp'], chunksize=chunk_size)):
    chunk['trackid_idx'] = chunk['track_id'].map(trackid_to_idx)
    mapped_chunks.append(chunk)

df_concatenated = pd.concat(mapped_chunks)
df_concatenated.to_parquet('plays.pqt')

