"""GNN (GCN / GraphSAGE) gene-prioritisation models and a 1-D CNN sample classifier, in plain PyTorch.

GCN layer:        H^{l+1} = σ(Â H^l W^l)
GraphSAGE (mean): h_v^{l+1} = σ(W_1 h_v^l + W_2 mean_{u∈N(v)} h_u^l)
Loss: class-weighted binary cross-entropy  L = -Σ_i [w_+ y_i log σ(z_i) + (1-y_i) log(1-σ(z_i))]."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_torch_sparse(M: sp.csr_matrix) -> torch.Tensor:
    M = M.tocoo()
    i = torch.tensor(np.vstack([M.row, M.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(i, torch.tensor(M.data, dtype=torch.float32), M.shape).coalesce().to_sparse_csr()


class GCN(nn.Module):
    def __init__(self, d_in, d_h=32, dropout=0.3):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h), nn.Linear(d_h, 1)
        self.dp = dropout

    def forward(self, x, Ahat):
        h = F.relu(torch.sparse.mm(Ahat, self.l1(x)))
        h = F.dropout(h, self.dp, self.training)
        h = F.relu(torch.sparse.mm(Ahat, self.l2(h)))
        return self.out(h).squeeze(-1)


class SAGE(nn.Module):
    def __init__(self, d_in, d_h=32, dropout=0.3):
        super().__init__()
        self.s1, self.n1 = nn.Linear(d_in, d_h), nn.Linear(d_in, d_h, bias=False)
        self.s2, self.n2 = nn.Linear(d_h, d_h), nn.Linear(d_h, d_h, bias=False)
        self.out = nn.Linear(d_h, 1)
        self.dp = dropout

    def forward(self, x, Amean):
        h = F.relu(self.s1(x) + self.n1(torch.sparse.mm(Amean, x)))
        h = F.dropout(h, self.dp, self.training)
        h = F.relu(self.s2(h) + self.n2(torch.sparse.mm(Amean, h)))
        return self.out(h).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, d_in, d_h=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_h, d_h), nn.ReLU(), nn.Linear(d_h, 1))

    def forward(self, x, _A=None):
        return self.net(x).squeeze(-1)


def seeded_node_model(model_type, *args, seed=0, **kwargs):
    """Seed before parameter initialization, not only before optimization.

    Forked RNG state avoids changing the caller's random stream.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return model_type(*args, **kwargs)


def train_node_model(model, x, A, y, train_mask, epochs=100, lr=0.01, wd=5e-4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    yt = torch.tensor(y, dtype=torch.float32)
    m = torch.tensor(train_mask)
    pos = max(1.0, float(yt[m].sum()))
    pw = torch.tensor((m.sum().item() - pos) / pos)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        z = model(x, A)
        loss = F.binary_cross_entropy_with_logits(z[m], yt[m], pos_weight=pw)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, A)).numpy()


class CNN1D(nn.Module):
    """Sample classifier over PPI-seriated gene vectors: conv(k=9) → pool → conv → global max → linear."""
    def __init__(self, n_genes, ch=16, dropout=0.3):
        super().__init__()
        self.c1 = nn.Conv1d(1, ch, 9, padding=4)
        self.c2 = nn.Conv1d(ch, ch, 9, padding=4)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(ch, 1)

    def forward(self, x):  # x: (B, G)
        h = F.relu(self.c1(x.unsqueeze(1)))
        h = F.max_pool1d(h, 4)
        h = F.relu(self.c2(h))
        h = torch.amax(h, dim=-1)
        return self.fc(self.dp(h)).squeeze(-1)


def train_cnn(Xtr, ytr, epochs=60, lr=3e-3, seed=0, batch=32):
    # Seed model weights before instantiation, not just the later minibatch order.
    net = seeded_node_model(CNN1D, Xtr.shape[1], seed=seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.tensor(Xtr, dtype=torch.float32); y = torch.tensor(ytr, dtype=torch.float32)
    pos = max(1.0, float(y.sum())); pw = torch.tensor((len(y) - pos) / pos)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(net(X[b]), y[b], pos_weight=pw).backward()
            opt.step()
    net.eval()
    return net


def predict_cnn(net, X):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32))).numpy()
