import soundfile as sf
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import torch
from musicfm.model.musicfm_25hz import MusicFM25Hz

HOME_PATH = '/gpfs/space/home/yanmart/music-hybrid'
sys.path.append(HOME_PATH)

track_ids = pd.read_csv('/gpfs/space/projects/music_ca/DATA/music4all/trackid_sorted.csv', index_col=0)['trackid']
track_paths = [ '/gpfs/space/projects/music_ca/DATA/music4all/wav24k/' + track_id + '.wav' for track_id in track_ids]

def read_mono(path, target_length=720000):
    audio = sf.read(path, stop=target_length)[0]
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    current_length = audio.shape[0]
    
    if current_length < target_length:
        silence_length = target_length - current_length
        silence = np.zeros(silence_length)
        audio = np.concatenate((audio, silence))
    return audio

def load_batch(track_paths, batch_size, index):
    tracks = track_paths[index * batch_size: (index + 1) * batch_size]
    embs = np.stack([read_mono(track) for track in tracks])
    return embs

def embed_batch(batch):
    with torch.no_grad():
        batch = torch.tensor(batch).cuda()
        emb = musicfm.get_latent(batch, layer_ix=12).mean(-1).cpu().numpy() # (batch, time, channel) -> (batch, channel)
        return emb

# msd, fma
dataset = "fma"

musicfm = MusicFM25Hz(
    is_flash=False,
    stat_path=os.path.join(HOME_PATH, "extract_item_embeddings", "musicfm", "data", f"{dataset}_stats.json"),
    model_path=os.path.join(HOME_PATH, "extract_item_embeddings", "musicfm", "data", f"pretrained_{dataset}.pt"),
).cuda()

musicfm.eval()



res = []
batch_size = 16
for i in tqdm(range(int(len(track_paths) / batch_size))):
    b = load_batch(track_paths, batch_size, i)
    reps = embed_batch(b)
    res.append(reps)

res = np.concatenate(res)
np.save(os.path.join(HOME_PATH, f'embeddings/musicfm_{dataset}.npy'), res)





