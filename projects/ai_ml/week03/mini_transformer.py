import torch
import torch.nn as nn
import math

torch.manual_seed(42)

# -------------------------
# 1. Self-Attention
# -------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V
        return out


# -------------------------
# 2. Transformer Block
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attn = SelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention + Residual
        x = x + self.attn(x)
        x = self.norm1(x)

        # MLP + Residual
        x = x + self.mlp(x)
        x = self.norm2(x)

        return x


# -------------------------
# 3. Test it
# -------------------------
batch_size = 2
seq_len = 5
d_model = 8

x = torch.randn(batch_size, seq_len, d_model)

block = TransformerBlock(d_model)
out = block(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)
