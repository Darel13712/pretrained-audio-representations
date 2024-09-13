from musicnn.extractor import extractor
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm


track_ids = pd.read_csv('/gpfs/space/projects/music_ca/DATA/music4all/trackid_sorted.csv', index_col=0)['trackid']
track_paths = [ '/gpfs/space/projects/music_ca/DATA/music4all/audio/' + track_id + '.mp3' for track_id in track_ids]

def get_embedding(file_path):
    try:
        _, _, features = extractor(file_path, model='MSD_musicnn', extract_features=True)
        features = features['penultimate'].mean(axis=0)
        return features
    except Exception as e:
        print(f"Failed for {file_path}: {str(e)}")
        return None



with Pool(30) as p:
    res = np.stack(p.map(get_embedding, track_paths))


np.save('~/music-hybrid/embeddings/musicnn.npy', res)
