import pandas as pd
import numpy as np
from tqdm import tqdm
from rs_metrics import hitrate, mrr, precision, recall, ndcg


train = pd.read_parquet('data/train.pqt')
val = pd.read_parquet('data/val.pqt')
pred = pd.read_parquet('recs/als.pqt')

pdict = pred.set_index('user_id').to_dict()['item_id']

hitrate(val, pdict, k=10, item_col='track_id')
mrr(val, pdict, k=10, item_col='track_id')
precision(val, pdict, k=10, item_col='track_id')
recall(val, pdict, k=10, item_col='track_id')
ndcg(val, pdict, k=10, item_col='track_id')

results = pd.DataFrame()
results.at['ALS', 'Hitrate@10'] = hitrate(val, pdict, k=10, item_col='track_id')
results.at['ALS', 'MRR@50'] = mrr(val, pdict, k=100, item_col='track_id')
results.at['ALS', 'Precision@50'] = precision(val, pdict, k=10, item_col='track_id')
results.at['ALS', 'Recall@50'] = recall(val, pdict, k=50, item_col='track_id')
results.at['ALS', 'NDCG@50'] = ndcg(val, pdict, k=50, item_col='track_id')
