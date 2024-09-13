from encodecmae import load_model
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm



model = load_model('base', device='cuda:0')

track_ids = pd.read_csv('/gpfs/space/projects/music_ca/DATA/music4all/trackid_sorted.csv', index_col=0)['trackid']
track_paths = [ '/gpfs/space/projects/music_ca/DATA/music4all/wav24k/' + track_id + '.wav' for track_id in track_ids]

def get_embedding(file_path):
    try:
        features = model.extract_features_from_file(file_path).mean(axis=0)
        return features
    except Exception as e:
        print(f"Failed for {file_path}: {str(e)}")
        return None


res = []
for p in tqdm(track_paths):
    res.append(get_embedding(p))
res = np.stack(res)

np.save('embeddings/encodecmae.npy', res)
