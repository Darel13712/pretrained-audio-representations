import sys
import os
import torch
import pandas as pd
import numpy as np
import faiss
import scipy
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datetime import datetime
from rs_metrics import hitrate, mrr, precision, recall, ndcg

from bert4rec import BERT4Rec
from dataset import InteractionDataset, InteractionDatasetItems

class LMDataset(Dataset):

    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=True,
                 user_col='user_id', item_col='item_id', time_col='timestamp'):

        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()

    def __len__(self):

        return len(self.data)

    def sample_negatives(self, item_sequence):

        negatives = self.all_items[~np.isin(self.all_items, item_sequence)]
        if self.full_negative_sampling:
            negatives = np.random.choice(
                negatives, size=self.num_negatives * (len(item_sequence) - 1), replace=True)
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        else:
            negatives = np.random.choice(negatives, size=self.num_negatives, replace=False)

        return negatives

class MaskedLMDataset(LMDataset):

    def __init__(self, df, max_length=128,
                 num_negatives=None, full_negative_sampling=True,
                 mlm_probability=0.2,
                 masking_value=1, ignore_value=-100,
                 force_last_item_masking_prob=0,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.mlm_probability = mlm_probability
        self.masking_value = masking_value
        self.ignore_value = ignore_value
        self.force_last_item_masking_prob = force_last_item_masking_prob

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length:
            item_sequence = item_sequence[-self.max_length:]

        input_ids = np.array(item_sequence)
        mask = np.random.rand(len(item_sequence)) < self.mlm_probability
        input_ids[mask] = self.masking_value
        if self.force_last_item_masking_prob > 0:
            if np.random.rand() < self.force_last_item_masking_prob:
                input_ids[-1] = self.masking_value

        labels = np.array(item_sequence)
        labels[input_ids != self.masking_value] = self.ignore_value

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class MaskedLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, masking_value=1,
                 validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.masking_value = masking_value
        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length:-1]
            item_sequence = item_sequence[:-1]
        else:
            input_ids = item_sequence[-self.max_length + 1:]

        input_ids += [self.masking_value]

        if self.validation_mode:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence, 'target': target}
        else:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence}

class PaddingCollateFn:

    def __init__(self, padding_value=0, labels_padding_value=-100):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key == 'labels':
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                               padding_value=padding_value)

        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch

def hinge_loss(y_pos, y_neg, confidence, dlt=0.2):
    loss = dlt - y_pos + y_neg
    loss = torch.clamp(loss, min=0) * confidence
    return torch.mean(loss)

def save_model(model, path, epoch, optimizer, best_val_loss=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_val_loss,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss})")
    return model, optimizer, loss

def make_small_train(tr, te, em, num=100000):
    tr = tr.head(num)
    te = te[te.user_id.isin(tr.user_id.unique()) & te.track_id.isin(tr.track_id.unique())]
    le = LabelEncoder()
    tr['user_id'] = le.fit_transform(tr['user_id'])
    te['user_id'] = le.transform(te['user_id'])
    tr['item_id'] = le.fit_transform(tr['track_id'])
    te['item_id'] = le.transform(te['track_id'])
    em = em[le.classes_]
    return tr, te, em

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

def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df


def main():
    print(sys.argv)
    # if len(sys.argv) != 9:
    print("Usage: python train.py $model $sample_type "
          "$numneg $batch $epochs $item_freeze "
          "$user_freeze $logdir")
    # sys.exit(1)

    model_name = sys.argv[1]
    sample_type = sys.argv[2]
    max_seq_len = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    num_epochs = int(sys.argv[5])
    item_freeze = bool(int(sys.argv[6]))
    user_freeze = bool(int(sys.argv[7]))
    comment = sys.argv[8]
    user_init = bool(int(sys.argv[9]))
    dynamic_item_freeze = bool(int(sys.argv[10]))
    hidden_dim = int(sys.argv[11])
    use_confidence = bool(int(sys.argv[12]))
    l2 = float(sys.argv[13])
    if len(sys.argv) > 14:
        logdir = sys.argv[14]
        last_epoch = int(sys.argv[15])
    else:
        logdir = None
        last_epoch = 0



    current_time = datetime.now()
    formatted_time = current_time.strftime("%b%d_%H:%M")
    run_name = f"bert_{model_name}-{formatted_time}_{hidden_dim}_{max_seq_len}_{comment}"
    if logdir is not None:
        run_name = logdir
    writer = SummaryWriter(log_dir='runs/' + run_name)

    writer.add_text('Params', f'Type: {sample_type}\n'
                              f'max_seq_len: {max_seq_len}\n'
                              f'Batch size: {batch_size}\n'
                              f'Item freeze: {item_freeze}\n'
                              f'Hidden dim: {hidden_dim}\n'
                              f'User freeze: {user_freeze}', global_step=0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = pd.read_parquet('data/train.pqt').sort_values(['user_id', 'timestamp'])
    val = pd.read_parquet('data/val.pqt').sort_values(['user_id', 'timestamp'])
    test = pd.read_parquet('data/test.pqt').sort_values(['user_id', 'timestamp'])

    if model_name != "random":
        embs = np.load(f'embeddings/{model_name}.npy')
        hidden_dim = embs.shape[1]
    else:
        embs = None

    # if hidden_dim == 0:
    #     hidden_dim = emb_dim_in

    # train, val, embs = make_small_train(train, val, embs, 50000)


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
    if embs is not None:
        embs = embs[np.sort(train.track_id.unique())]
        special_embs = np.random.normal(loc=0.0, scale=0.02, size=(2, hidden_dim))
        embs = np.concatenate([embs, special_embs], axis=0)


    user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()
    vc = val.groupby('user_id')['item_id'].count()
    val = val[val['user_id'].isin(vc[vc >= 10].index)]

    item_count = train.item_id.nunique() + 2


    train_dataset = MaskedLMDataset(train, masking_value=item_count-2, max_length=max_seq_len, mlm_probability=0.2, force_last_item_masking_prob=0)
    eval_dataset = MaskedLMDataset(val, masking_value=item_count-2, max_length=max_seq_len, mlm_probability=0.2, force_last_item_masking_prob=0)
    pred_dataset = MaskedLMPredictionDataset(train, masking_value=item_count-2, max_length=50, validation_mode=False)


    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=2,
                              collate_fn=PaddingCollateFn(padding_value=item_count-1))
    eval_loader = DataLoader(eval_dataset, batch_size=128,
                             shuffle=False, num_workers=2,
                             collate_fn=PaddingCollateFn(padding_value=item_count-1))
    pred_loader = DataLoader(pred_dataset, batch_size=128,
                             shuffle=False, num_workers=2,
                             collate_fn=PaddingCollateFn(padding_value=item_count-1))


    model_params = {
        'vocab_size': 2,
        'max_position_embeddings': max(200, max_seq_len),
        'hidden_size': hidden_dim,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256
    }
    model = BERT4Rec(vocab_size=item_count, add_head=True, precomputed_item_embeddings=embs, padding_idx=item_count-1,
                     bert_config=model_params)

    model.freeze_item_embs(item_freeze)



    model.to(device)

    patience_counter = 0
    patience_threshold = 16

    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

    if last_epoch > 0:
        model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_{last_epoch - 1}.pt')
        best_val_loss = torch.load(f'checkpoints/{model_name}/{run_name}_best.pt')['loss']

    for epoch in tqdm(range(last_epoch, num_epochs)):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()


        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        # save_model(model, f'checkpoints/{model_name}/{run_name}_{epoch}.pt', epoch, optimizer)

        # if epoch % 5 == 0 and epoch > 0:
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(eval_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                print(f'New best at epoch {epoch}')
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_model(model, f'checkpoints/{model_name}/{run_name}_best.pt', epoch, optimizer, best_val_loss)
            else:
                patience_counter += 1
            if patience_counter >= patience_threshold:
                print('Applying early stop')
                break

    model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_best.pt')
    # user_embs, item_embs = model.extract_embeddings()


    user_recommendations = {}
    k = 100

    with torch.no_grad():
        model.eval()
        for batch in tqdm(pred_loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_ids = batch['user_id']

            outputs = model(input_ids, attention_mask)

            seq_lengths = attention_mask.sum(dim=1).long()

            last_item_logits = torch.stack([outputs[i, seq_lengths[i] - 1, :] for i in range(len(seq_lengths))])
            last_item_logits = last_item_logits[:, :-2] # remove mask and padding tokens
            scores, preds = torch.sort(last_item_logits, descending=True)
            preds = preds.cpu().numpy()

            for user_id, item_ids in zip(user_ids, preds):
                user_id = user_id.item()
                history = user_history[user_id]
                recs = [item_id for item_id in item_ids if item_id not in history][:k]
                user_recommendations[user_id] = recs


    df = dict_to_pandas(user_recommendations)
    # df['item_id'] = ie.inverse_transform(df['item_id'])
    # df['user_id'] = ue.inverse_transform(df['user_id'])
    df.to_parquet(f'preds/{run_name}.pqt')

    metrics_val = calc_metrics(val, df)
    metrics_val.to_parquet(f'metrics/{run_name}_val.pqt')
    metrics_val = metrics_val.apply(mean_confidence_interval)
    metrics_val.index = ['mean', 'conf']
    metrics_val.to_csv(f'metrics/{run_name}_val.csv')
    print('Val metrics:')
    print(metrics_val)

    metrics_test = calc_metrics(test, df)
    metrics_test.to_parquet(f'metrics/{run_name}_test.pqt')
    metrics_test = metrics_test.apply(mean_confidence_interval)
    metrics_test.index = ['mean', 'conf']
    metrics_test.to_csv(f'metrics/{run_name}_test.csv')
    print('Test metrics:')
    print(metrics_test)

    for metric_name, metric_value in metrics_val.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    for metric_name, metric_value in metrics_test.items():
        writer.add_scalar(f'Test/{metric_name}', metric_value['mean'], 0)

    writer.close()

if __name__ == '__main__':
    main()
