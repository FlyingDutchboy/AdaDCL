import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
from Args import args
from DataHandler import DataHandler
import torch.nn.functional as F
from utility import setup_seed
# ================= Configuration =================
LR = 0.001
DECAY = 1e-4
LAYER_NUM = 3
gamma = 2


# ===========================================

class LightGCN_Encoder(nn.Module):
    def __init__(self, n_user, n_item, emb_size, n_layers, norm_adj):
        super(LightGCN_Encoder, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = norm_adj  # Sparse adjacency matrix

        # Initialize Embeddings
        self.user_emb = nn.Embedding(n_user, emb_size)
        self.item_emb = nn.Embedding(n_item, emb_size)
        # Use Xavier initialization
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self):
        # Concatenate user and item embeddings
        ego_embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embeddings = [ego_embeddings]

        # Graph Convolution Propagation
        for k in range(self.n_layers):
            # E(k+1) = L * E(k)
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Mean Pooling
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)

        # Split back into users and items
        users_emb, items_emb = torch.split(final_embeddings, [self.n_user, self.n_item])
        return users_emb, items_emb


def bpr_loss(users, pos_items, neg_items):
    pos_scores = torch.sum(users * pos_items, dim=1)
    neg_scores = torch.sum(users * neg_items, dim=1)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    return loss

def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()
def uniformity(x,t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
def calculate_loss(user_emb,item_emb):
    align = alignment(user_emb,item_emb)
    uniform = gamma * (uniformity(user_emb) + uniformity(item_emb)) / 2
    return  align + uniform
#

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2).pow(2)
    return emb_loss * reg


def main():
    setup_seed(args.random_seed)
    # 1. Set Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device:{args.cuda}")
    print(f"Dataset: {args.data}")

    # 2. Load Data
    print("Loading DataHandler...")
    # DataHandler loads data and builds the graph in __init__
    handler = DataHandler()

    # Use handler.graph directly (it is already a sparse tensor on the correct device or CPU)
    # We ensure it is on the correct device
    adj = handler.graph.to(device)

    # 3. Initialize Model
    print("Initializing LightGCN...")
    model = LightGCN_Encoder(
        n_user=handler.n_user,
        n_item=handler.n_item,
        emb_size=args.hidden_factor,
        n_layers=LAYER_NUM,
        norm_adj=adj
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    print(f"Start Pre-training for {args.pretrain_epochs} epochs...")
    trnLoader = handler.trnLoader

    for epoch in tqdm(range(args.pretrain_epochs)):
        total_loss = 0
        t_start = time.time()

        for i, batch in tqdm(enumerate(trnLoader)):
            # Get batch data
            user_idx, pos_idx = batch
            user_idx = user_idx.long().to(device)
            pos_idx = pos_idx.long().to(device)

            # Negative Sampling
            neg_idx = trnLoader.dataset.negSampling(pos_idx, user_idx)
            neg_idx = torch.tensor(neg_idx).long().to(device)

            # Forward Pass
            all_users_emb, all_items_emb = model()

            u_e = all_users_emb[user_idx]
            pos_e = all_items_emb[pos_idx]
            neg_e = all_items_emb[neg_idx]

            # Base Embeddings for Regularization
            u_e_0 = model.user_emb(user_idx)
            pos_e_0 = model.item_emb(pos_idx)
            neg_e_0 = model.item_emb(neg_idx)
            loss_au = calculate_loss(u_e_0, pos_e_0)
            # Calculate Loss
            loss = (bpr_loss(u_e, pos_e, neg_e)
                    + l2_reg_loss(DECAY, u_e_0, pos_e_0, neg_e_0) / user_idx.shape[0]
                    + loss_au * 0.001)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.pretrain_epochs} | Loss: {total_loss:.4f} | Time: {time.time() - t_start:.2f}s")

    # 5. Save Embeddings as .npy
    print("Saving Embeddings...")
    save_dir = os.path.join(args.data_dir, args.data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        final_user_emb, final_item_emb = model()
        # Convert to numpy
        u_emb_np = final_user_emb.cpu().numpy()
        i_emb_np = final_item_emb.cpu().numpy()

        # Generate paths
        u_path = os.path.join(save_dir, "uEmbeds_pt.npy")
        i_path = os.path.join(save_dir, "iEmbeds_pt.npy")

        np.save(u_path, u_emb_np)
        np.save(i_path, i_emb_np)

    print(f"Successfully saved to:\n  {u_path}\n  {i_path}")
    print("Done. Now you can run the main AdaDCL code.")


if __name__ == "__main__":
    main()
