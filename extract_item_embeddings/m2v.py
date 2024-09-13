from transformers import AutoModel
import torch
import soundfile as sf
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd


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

def embed_batch(model, batch):
    with torch.no_grad():
        batch = torch.from_numpy(batch).float().cuda()
        emb = model(batch).last_hidden_state.mean(-2).cpu().numpy()
        return emb

def embed(model, track_paths, batch_size=16):
    assert len(track_paths) % batch_size == 0
    res = []
    for i in tqdm(range(int(len(track_paths) / batch_size))):
        b = load_batch(track_paths, batch_size, i)
        reps = embed_batch(model, b)
        res.append(reps)

    res = np.concatenate(res)
    return res


model = AutoModel.from_pretrained("m-a-p/music2vec-v1", trust_remote_code=True).cuda()
e = embed(model, track_paths, 16)
np.save('embeddings/music2vec.npy', e)




