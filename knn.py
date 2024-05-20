import sys
import os
import torch
import pandas as pd
import numpy as np
import faiss
import scipy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from rs_metrics import hitrate, mrr, precision, recall, ndcg

train = pd.read_parquet('data/train.pqt')
val = pd.read_parquet('data/val.pqt')
test = pd.read_parquet('data/test.pqt')

train['item_id'] = train['track_id']
val['item_id'] = val['track_id']
test['item_id'] = test['track_id']

ue = LabelEncoder()
ie = LabelEncoder()
train['user_id'] = ue.fit_transform(train['user_id'])
train['item_id'] = ie.fit_transform(train['item_id'])
val['user_id'] = ue.transform(val['user_id'])
val['item_id'] = ie.transform(val['item_id'])
test['user_id'] = ue.transform(test['user_id'])
test['item_id'] = ie.transform(test['item_id'])

user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()
k = 100

def dict_to_pandas(d, key_col='user_id', val_col='item_id'):
    return (
        pd.DataFrame(d.items(), columns=[key_col, val_col])
            .explode(val_col)
            .reset_index(drop=True)
    )

def calc_metrics(test, pred, k=50):
    metrics = pd.DataFrame()
    metrics[f'HitRate@{k}'] = hitrate(test, pred, k=50, apply_mean=False)
    metrics[f'MRR@{k}'] = mrr(test, pred, k=50, apply_mean=False)
    metrics[f'Precision@{k}'] = precision(test, pred, k=50, apply_mean=False)
    metrics[f'Recall@{k}'] = recall(test, pred, k=50, apply_mean=False)
    metrics[f'NDCG@{k}'] = ndcg(test, pred, k=50, apply_mean=False)
    return metrics

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def write_metrics(run_name):
    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    df = pd.read_parquet(f'metrics/{run_name}_val.pqt')
    df = df.apply(mean_confidence_interval)
    df.index = ['mean', 'conf']
    for metric_name, metric_value in df.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    df = pd.read_parquet(f'metrics/{run_name}_test.pqt')
    df = df.apply(mean_confidence_interval)
    df.index = ['mean', 'conf']
    for metric_name, metric_value in df.items():
        writer.add_scalar(f'Test/{metric_name}', metric_value['mean'], 0)

    writer.close()


def calc_knn(model_name, suffix="cosine"):
    run_name = f'{model_name}_{suffix}'
    print(run_name)
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    item_embs = np.load(f'embeddings/{model_name}.npy')
    item_embs = item_embs[np.sort(train.track_id.unique())]
    user_embs = np.stack(train.groupby('user_id')['item_id'].apply(lambda items: item_embs[items].mean(axis=0)).values)
    if suffix == 'cosine':
        user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
        item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    all_users = np.concatenate((val.user_id.unique(), test.user_id.unique()))
    user_recommendations = {}
    for user_id in tqdm(all_users):
        history = user_history[user_id]
        user_vector = user_embs[user_id]
        distances, indices = index.search(np.array([user_vector]), k + len(history))
        recommendations = [idx for idx in indices[0] if idx not in history][:k]
        user_recommendations[user_id] = recommendations
    df = dict_to_pandas(user_recommendations)

    metrics_val = calc_metrics(val, df)
    metrics_val.to_parquet(f'metrics/{run_name}_val.pqt')
    metrics_val = metrics_val.apply(mean_confidence_interval)
    metrics_val.index = ['mean', 'conf']
    metrics_val.to_csv(f'metrics/{run_name}_val.csv')
    print('Val metrics:')
    print(metrics_val)
    for metric_name, metric_value in metrics_val.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    metrics_test = calc_metrics(test, df)
    metrics_test.to_parquet(f'metrics/{run_name}_test.pqt')
    metrics_test = metrics_test.apply(mean_confidence_interval)
    metrics_test.index = ['mean', 'conf']
    metrics_test.to_csv(f'metrics/{run_name}_test.csv')
    print('Test metrics:')
    print(metrics_test)

    for metric_name, metric_value in metrics_test.items():
        writer.add_scalar(f'Test/{metric_name}', metric_value['mean'], 0)

    writer.close()

for model in ['musicnn', 'encodecmae', 'jukemir', 'music2vec', 'mert', 'musicfm', 'mfcc', 'mfcc_bow']:
    calc_knn(model)