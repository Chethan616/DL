import os, sys, math, json, random, time, copy, warnings
from pathlib import Path
from itertools import product
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_POINTS  = 256
MAX_HISTORY = 12

class IntraCycleEncoder(nn.Module):
    def __init__(self, input_features=5, hidden_size=128, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_features, hidden_size), nn.GELU(),
            nn.LayerNorm(hidden_size), nn.Dropout(dropout))
    def forward(self, x, point_mask):
        B, H, P, F = x.shape
        flat_x = x.reshape(B * H, P, F)
        encoded = self.projection(flat_x)
        mask = point_mask.reshape(B * H, P).bool().unsqueeze(-1)
        denom = mask.float().sum(dim=1).clamp_min(1)
        mean = (encoded * mask.float()).sum(dim=1) / denom
        max_val = encoded.masked_fill(~mask, -1e4).max(dim=1).values
        pooled = 0.5 * (mean + max_val)
        return pooled.reshape(B, H, -1)

class HealthHeads(nn.Module):
    def __init__(self, hidden_size=128, rul_scale=250.0):
        super().__init__()
        self.soh_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.GELU(), nn.Linear(hidden_size//2, 1))
        self.rul_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.GELU(), nn.Linear(hidden_size//2, 1))
        self.rul_scale = float(rul_scale)
    def forward(self, seq):
        soh = 100.0 * torch.sigmoid(self.soh_head(seq).squeeze(-1))
        rul = self.rul_scale * torch.sigmoid(self.rul_head(seq).squeeze(-1))
        return {'soh': soh, 'rul': rul, 'embedding': seq}

class FeedForwardTwin(nn.Module):
    def __init__(self, input_features=5, hidden_size=128, dropout=0.1, rul_scale=250.0, **kw):
        super().__init__()
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size*2, hidden_size))
        self.output_heads = HealthHeads(hidden_size, rul_scale)
    def forward(self, x, pm, cm): return self.output_heads(self.ffn(self.cycle_encoder(x, pm)))

class GRUBatteryTwin(nn.Module):
    def __init__(self, input_features=5, hidden_size=128, dropout=0.1, rul_scale=250.0, num_layers=2, **kw):
        super().__init__()
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_heads = HealthHeads(hidden_size, rul_scale)
    def forward(self, x, pm, cm):
        seq, _ = self.temporal_encoder(self.cycle_encoder(x, pm))
        return self.output_heads(seq)

class TransformerBatteryTwin(nn.Module):
    def __init__(self, input_features=5, hidden_size=128, num_heads=4, num_layers=2, dropout=0.1, rul_scale=250.0, max_history=12, **kw):
        super().__init__()
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_history, hidden_size))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_heads = HealthHeads(hidden_size, rul_scale)
    def forward(self, x, pm, cm):
        c_emb = self.cycle_encoder(x, pm) + self.pos_embed[:, :x.shape[1]]
        seq = self.temporal_encoder(c_emb, src_key_padding_mask=~cm.bool())
        return self.output_heads(seq)

class PINNBatteryTwin(TransformerBatteryTwin):
    pass  # Same architecture, physics loss applied during training

class AdaptiveBatteryTwin(nn.Module):
    def __init__(self, input_features=5, hidden_size=128, num_heads=4, num_layers=2, dropout=0.1, rul_scale=250.0, max_history=12, **kw):
        super().__init__()
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_history, hidden_size))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_heads = HealthHeads(hidden_size, rul_scale)
        self.replay_buffer   = deque(maxlen=128)
        self.cusum_sum       = 0.0
        self.cusum_threshold = 2.5
        self.cusum_k         = 0.5
    def forward(self, x, pm, cm):
        c_emb = self.cycle_encoder(x, pm) + self.pos_embed[:, :x.shape[1]]
        seq = self.temporal_encoder(c_emb, src_key_padding_mask=~cm.bool())
        return self.output_heads(seq)

MODEL_CLASSES = {
    'dnn': FeedForwardTwin, 'gru': GRUBatteryTwin,
    'transformer': TransformerBatteryTwin, 'pinn': PINNBatteryTwin,
    'adaptive': AdaptiveBatteryTwin,
}

dummy_x  = torch.zeros(2, MAX_HISTORY, MAX_POINTS, 5).to(DEVICE)
dummy_pm = torch.ones(2, MAX_HISTORY, MAX_POINTS).to(DEVICE)
dummy_cm = torch.ones(2, MAX_HISTORY).to(DEVICE)

print(f"Device: {DEVICE}")
for name, cls in MODEL_CLASSES.items():
    m = cls(max_history=MAX_HISTORY).to(DEVICE)
    out = m(dummy_x, dummy_pm, dummy_cm)
    params = sum(p.numel() for p in m.parameters())
    soh_shape = out['soh'].shape
    rul_shape = out['rul'].shape
    print(f"  {name:15s} | soh={soh_shape} | rul={rul_shape} | params={params:,}")

print("\nAll 5 model architectures: OK!")
