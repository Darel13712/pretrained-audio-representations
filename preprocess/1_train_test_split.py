import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

START_DATE = '2019-02-20'
TEST_DATE = '2020-02-20'
# MAX_DATE = '2020-03-20'

compress = lambda x: x.groupby(['user_id', 'track_id']).agg(timestamp=("timestamp", "min"), count=("timestamp", "count")).reset_index()

df = pd.read_parquet('../m4a/plays.pqt')

df = df[df.timestamp >= pd.to_datetime(START_DATE)]
a = compress(df)

te = a[a.timestamp >= pd.to_datetime(TEST_DATE)]

# This is done so that the number of plays in train doesn't account for plays that happened after TEST_DATE
df = df[df.timestamp < pd.to_datetime(TEST_DATE)]
tr = compress(df)

te = te[te.user_id.isin(tr.user_id.unique())]
te = te[te.track_id.isin(tr.track_id.unique())]


validation_user_ids, test_user_ids = train_test_split(te.user_id.unique(), test_size=0.5, random_state=42)
val = te[te.user_id.isin(validation_user_ids)]
test = te[te.user_id.isin(test_user_ids)]

tr.to_parquet('data/train.pqt')
val.to_parquet('data/val.pqt')
test.to_parquet('data/test.pqt')

