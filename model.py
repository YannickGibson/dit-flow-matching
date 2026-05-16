"""DiT (Diffusion Transformer) for 32x32x1 FashionMNIST.

Faithful to "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022):
patchify -> N transformer blocks with adaLN-zero conditioning -> linear head.
Trained here with a flow-matching objective (see flow.py), so the network
predicts a velocity field rather than DDPM noise.
"""

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Positional embedding (fixed 2D sin-cos, as in the DiT paper)
# ----------------------------------------------------------------------------
def get_1d_sincos(dim, pos):
    omega = np.arange(dim // 2) / (dim / 2.0)
    omega = 1.0 / 10000**omega
    out = pos.reshape(-1)[:, None] * omega[None, :]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(dim, grid_size):
    grid = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), axis=0)
    grid = grid.reshape(2, -1)
    emb_h = get_1d_sincos(dim // 2, grid[1])
    emb_w = get_1d_sincos(dim // 2, grid[0])
    return np.concatenate([emb_h, emb_w], axis=1)  # (grid_size**2, dim)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ----------------------------------------------------------------------------
# Conditioning embedders
# ----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """Embeds a continuous flow-matching time t in [0, 1]."""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    """Class embedding with classifier-free-guidance dropout.

    Index `num_classes` is the reserved 'null' / unconditional token.
    """

    def __init__(self, num_classes, hidden, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop, self.num_classes, labels)
        return self.embedding_table(labels)


# ----------------------------------------------------------------------------
# DiT block with adaLN-zero
# ----------------------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        # Produces 6 modulation signals: shift/scale/gate for attn and mlp.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch_size, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ----------------------------------------------------------------------------
# Full DiT
# ----------------------------------------------------------------------------
class DiT(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_ch=1, hidden=256,
                 depth=8, heads=4, num_classes=10, cfg_dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        num_patches = self.grid**2

        self.x_embed = nn.Conv2d(in_ch, hidden, patch_size, stride=patch_size)
        self.t_embed = TimestepEmbedder(hidden)
        self.y_embed = LabelEmbedder(num_classes, hidden, cfg_dropout)

        pos = get_2d_sincos_pos_embed(hidden, self.grid)
        self.register_buffer("pos_embed", torch.tensor(pos, dtype=torch.float32)[None])

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads) for _ in range(depth)]
        )
        self.final = FinalLayer(hidden, patch_size, in_ch)
        self._init_weights()

    def _init_weights(self):
        # adaLN-zero: zero the last layer of every modulation MLP so each
        # block / the final layer starts as an identity mapping.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        b = x.shape[0]
        p, g, c = self.patch_size, self.grid, self.in_ch
        x = x.reshape(b, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(self, x, t, y, train=False):
        x = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embed(t) + self.y_embed(y, train)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final(x, c))
