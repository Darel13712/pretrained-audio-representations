import jukemirlib
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

track_ids = pd.read_csv('/gpfs/space/projects/music_ca/DATA/music4all/trackid_sorted.csv', index_col=0)['trackid']
track_paths = [ '/gpfs/space/projects/music_ca/DATA/music4all/wav44k/' + track_id + '.wav' for track_id in track_ids]


res = []
for i in tqdm(range(int(len(track_paths) / 8))):
    f = track_paths[i * 8: (i + 1) * 8]
    reps = jukemirlib.extract(fpath=f, fp16=True, layers=[36], duration=24)[36].mean(axis=1)
    np.save(f'embeddings/jb/{i}.npy', reps)

