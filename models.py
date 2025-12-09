import torch
import torch.nn as nn
from einops import rearrange

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size-1)*dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)
        out = out[..., :x.size(-1)]
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class FinboostModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, levels=4, heads=4):
        super().__init__()
        layers = []
        ch = input_dim
        for i in range(levels):
            d = 2**i
            layers.append(TCNBlock(ch, hidden_dim, dilation=d))
            ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.attn = AttentionBlock(hidden_dim, heads=heads)
        self.shared = nn.Sequential(nn.Linear(hidden_dim,64), nn.ReLU(), nn.Dropout(0.2))
        self.next_candle = nn.Linear(64,1)
        self.reversal = nn.Linear(64,1)
        self.regime = nn.Linear(64,3)

    def forward(self, x):
        # x: batch, seq_len, features
        x = x.transpose(1,2)  # -> batch, features, seq
        x = self.tcn(x)
        x = x.transpose(1,2)  # -> batch, seq, ch
        x = self.attn(x)
        x = x[:,-1,:]
        z = self.shared(x)
        return {
            'next_candle': self.next_candle(z),
            'reversal': torch.sigmoid(self.reversal(z)),
            'regime': torch.softmax(self.regime(z), dim=-1)
        }
