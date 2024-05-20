import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ShallowEmbeddingModel(nn.Module):
    # todo: поменять на параметр users, items который либо инт либо нумпай
    def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, precomputed_user_embeddings=None, emb_dim_out=300):
        super(ShallowEmbeddingModel, self).__init__()
        self.emb_dim_in = emb_dim_in

        if precomputed_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in)
        else:
            precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings)
            assert precomputed_user_embeddings.size(1) == emb_dim_in
            self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings)

        if precomputed_item_embeddings is None:
            self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in)
        else:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings)
            assert precomputed_item_embeddings.size(1) == emb_dim_in
            self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings)

        self.model = nn.Sequential(
            nn.Linear(self.emb_dim_in, emb_dim_out),
            nn.ReLU()
        )

        # self.model = nn.Sequential(
        #     nn.Linear(self.emb_dim_in, self.emb_dim_in // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim_in // 2, emb_dim_out),
        #     nn.ReLU()
        # )

        self.cossim = torch.nn.CosineSimilarity()

    def freeze_item_embs(self, flag):
        self.item_embeddings.weight.requires_grad = flag

    def freeze_user_embs(self, flag):
        self.user_embeddings.weight.requires_grad = flag

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)

        user_embeds = self.model(user_embeds)
        item_embeds = self.model(item_embeds)

        scores = self.cossim(user_embeds, item_embeds)
        return scores

    def extract_embeddings(self):
        """
        Extract normalized embeddings
        """
        # Extract raw embeddings
        user_embeddings = self.user_embeddings.weight.data
        item_embeddings = self.item_embeddings.weight.data

        # Process through the model's sequential layers
        with torch.no_grad():
            user_embeddings = self.model(user_embeddings).cpu().numpy()
            item_embeddings = self.model(item_embeddings).cpu().numpy()

        user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        return user_embeddings, item_embeddings

class ShallowInteractionModel(nn.Module):
    # todo: поменять на параметр users, items который либо инт либо нумпай
    def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, precomputed_user_embeddings=None, emb_dim_out=300):
        super(ShallowInteractionModel, self).__init__()
        self.emb_dim_in = emb_dim_in

        if precomputed_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in)
        else:
            precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings)
            assert precomputed_user_embeddings.size(1) == emb_dim_in
            self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings)

        if precomputed_item_embeddings is None:
            self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in)
        else:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings)
            assert precomputed_item_embeddings.size(1) == emb_dim_in
            self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings)

        self.model = nn.Sequential(
            nn.Linear(self.emb_dim_in, emb_dim_out),
            nn.ReLU()
        )

        self.fc = nn.Linear(emb_dim_out, 1)
        # self.model = nn.Sequential(
        #     nn.Linear(self.emb_dim_in, self.emb_dim_in // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim_in // 2, emb_dim_out),
        #     nn.ReLU()
        # )

        # self.cossim = torch.nn.CosineSimilarity()

    def freeze_item_embs(self, flag):
        self.item_embeddings.weight.requires_grad = flag

    def freeze_user_embs(self, flag):
        self.user_embeddings.weight.requires_grad = flag

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)

        user_embeds = self.model(user_embeds)
        item_embeds = self.model(item_embeds)

        hadamard_product = user_embeds * item_embeds
        scores = self.fc(hadamard_product)

        return torch.sigmoid(scores).squeeze()

    def extract_embeddings(self):
        """
        Extract normalized embeddings
        """
        # Extract raw embeddings
        user_embeddings = self.user_embeddings.weight.data
        item_embeddings = self.item_embeddings.weight.data

        # Process through the model's sequential layers
        with torch.no_grad():
            user_embeddings = self.model(user_embeddings).cpu().numpy()
            item_embeddings = self.model(item_embeddings).cpu().numpy()

        user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        return user_embeddings, item_embeddings

