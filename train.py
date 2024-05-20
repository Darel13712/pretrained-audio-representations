import sys
import os
import torch
import pandas as pd
import numpy as np
import faiss
import scipy
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from rs_metrics import hitrate, mrr, precision, recall, ndcg

from dataset import InteractionDataset, InteractionDatasetItems
from model import ShallowEmbeddingModel


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


def main():
    print(sys.argv)
    # if len(sys.argv) != 9:
    print("Usage: python train.py $model $sample_type "
          "$numneg $batch $epochs $item_freeze "
          "$user_freeze $logdir")
        # sys.exit(1)

    model_name = sys.argv[1]
    sample_type = sys.argv[2]
    neg_samples = int(sys.argv[3])
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
    run_name = f"{model_name}-{formatted_time}_{hidden_dim}_{neg_samples}_{comment}"
    if logdir is not None:
        run_name = logdir
    writer = SummaryWriter(log_dir='runs/' + run_name)

    writer.add_text('Params', f'Type: {sample_type}\n'
                              f'Negatives: {neg_samples}\n'
                              f'Batch size: {batch_size}\n'
                              f'Item freeze: {item_freeze}\n'
                              f'Hidden dim: {hidden_dim}\n'
                              f'User freeze: {user_freeze}', global_step=0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/val.pqt')
    test = pd.read_parquet('data/test.pqt')

    if model_name == "random":
        embs = np.random.rand(56512, hidden_dim if hidden_dim != 0 else 64)
    else:
        embs = np.load(f'embeddings/{model_name}.npy')
    emb_dim_in = embs.shape[1]
    if hidden_dim == 0:
        hidden_dim = emb_dim_in

    # train, val, embs = make_small_train(train, val, embs, 50000)
    if model_name == "random":
        embs = None

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

    user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()

    user_embs = np.stack(train.groupby('user_id')['item_id'].apply(lambda items: embs[items].mean(axis=0)).values) if user_init else None

    Dataset = InteractionDataset if sample_type == "user" else InteractionDatasetItems

    train_dataset = Dataset(train, neg_samples=neg_samples)
    val_dataset = Dataset(val, neg_samples=neg_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ShallowEmbeddingModel(
        train.user_id.nunique(),
        train.item_id.nunique(),
        emb_dim_in,
        precomputed_item_embeddings=embs,
        precomputed_user_embeddings=user_embs,
        emb_dim_out=hidden_dim
    )
    model.freeze_item_embs(item_freeze)
    model.freeze_user_embs(user_freeze)

    model.to(device)

    patience_counter = 0
    patience_threshold = 16

    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

    if last_epoch > 0:
        model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_{last_epoch - 1}.pt')
        best_val_loss = torch.load(f'checkpoints/{model_name}/{run_name}_best.pt')['loss']

    for epoch in tqdm(range(last_epoch, num_epochs)):
        model.train()
        total_train_loss = 0

        for user, positive_item, confidence in train_loader:
            user = user.repeat_interleave(neg_samples).to(device)
            positive_item = positive_item.repeat_interleave(neg_samples).to(device)
            negative_items = torch.from_numpy(np.random.randint(0, 17052, len(user))).to(device)
            # negative_items = negative_items.view(-1).to(device)
            if not use_confidence:
                confidence = torch.tensor([1]*len(confidence))
            confidence = confidence.repeat_interleave(neg_samples).to(device)
            if use_confidence:
                confidence = (1 + 2 * torch.log(1 + confidence))
            optimizer.zero_grad()

            if sample_type == 'user':
                pos_score = model(user, positive_item)
                neg_scores = model(user, negative_items)
            elif sample_type == 'item': # In this case Dataset samples users instead of items
                pos_score = model(positive_item, user)
                neg_scores = model(negative_items, user)
            loss = hinge_loss(pos_score, neg_scores, confidence)
            if l2 > 0:
                l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
                loss = loss + l2 * l2_loss
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
            for user, positive_item, confidence in val_loader:
                user = user.repeat_interleave(neg_samples).to(device)
                positive_item = positive_item.repeat_interleave(neg_samples).to(device)
                negative_items = torch.from_numpy(np.random.randint(0, 17052, len(user))).to(device)
                # negative_items = negative_items.view(-1).to(device)
                if not use_confidence:
                    confidence = torch.tensor([1]*len(confidence))
                confidence = confidence.repeat_interleave(neg_samples).to(device)
                if use_confidence:
                    confidence = (1 + 2 * torch.log(1 + confidence))
                if sample_type == 'user':
                    pos_score = model(user, positive_item)
                    neg_scores = model(user, negative_items)
                elif sample_type == 'item': # In this case Dataset samples users instead of items
                    pos_score = model(positive_item, user)
                    neg_scores = model(negative_items, user)
                loss = hinge_loss(pos_score, neg_scores, confidence)
                if l2 > 0:
                    l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
                    loss = loss + l2 * l2_loss
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                print(f'new best at epoch {epoch}')
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_model(model, f'checkpoints/{model_name}/{run_name}_best.pt', epoch, optimizer, best_val_loss)
            else:
                patience_counter += 1
            if patience_counter == patience_threshold:
                print('Applying early stop')
                break

    model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_best.pt')
    user_embs, item_embs = model.extract_embeddings()
    np.save(f'model_embeddings/{run_name}_users.npy', user_embs)
    np.save(f'model_embeddings/{run_name}_items.npy', item_embs)

    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)

    user_recommendations = {}
    k = 100

    all_users = np.concatenate((val.user_id.unique(), test.user_id.unique()))
    for user_id in tqdm(all_users):
        history = user_history[user_id]
        user_vector = user_embs[user_id]
        distances, indices = index.search(np.array([user_vector]), k + len(history))
        recommendations = [idx for idx in indices[0] if idx not in history][:k]
        user_recommendations[user_id] = recommendations

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
