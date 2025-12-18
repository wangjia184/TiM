"""
model.py — TiM-style C2I model in **pixel space** (64×64 RGB), with **global attributes** conditioning.

Key differences from the paper repo:
- No VAE: we directly model RGB in [-1, 1].
- No CFG: conditioning (global_attrs) is always provided.

Key alignment with the paper repo:
- Model signature matches TransitionSchedule usage: forward(x_t, t_input, r_input, global_attrs, ...)
- Patchify + transformer + unpatchify structure.
- TiM-style conditioning via adaLN-Zero (simplified).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, N, D], shift/scale: [B, 1, D]
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Same positional embedding style as reference TiM (`tim/models/c2i/tim_model.py`)."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def positional_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        return self.mlp(t_freq)


class PatchEmbed(nn.Module):
    """Minimal patch embedding (Conv2d with stride=patch_size)."""
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B, N, D]
        x = self.proj(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class AdaLNZeroBlock(nn.Module):
    """TiM-style Transformer block with adaLN-Zero conditioning (simplified)."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        # produce: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # zero-init the last layer (adaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D], c: [B,1,D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * h
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class TiMC2IModel(nn.Module):
    """
    Pixel-space TiM-like model that predicts F(x_t, t, r, attrs) with the same shape as x_t.
    - x_t: [B, 3, 64, 64] in [-1, 1] (no VAE)
    - t_input, r_input: [B] (already transformed by transport.c_noise)
    - global_attrs: [B, attr_dim]
    """
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_chans: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attr_dim: int = 8,
        new_condition: str = "t-r",  # aligned with reference options
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.new_condition = new_condition

        self.x_embedder = PatchEmbed(img_size, patch_size, in_chans, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.delta_embedder = TimestepEmbedder(hidden_size)
        self.attr_embedder = nn.Linear(attr_dim, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=in_chans)

        # init patch embed like linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.x_embedder.proj.bias)
        # attr embed init
        nn.init.xavier_uniform_(self.attr_embedder.weight)
        nn.init.zeros_(self.attr_embedder.bias)

    def _get_delta_embed(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.new_condition == "t-r":
            return self.delta_embedder(t - r)
        if self.new_condition == "r":
            return self.delta_embedder(r)
        if self.new_condition == "t,r":
            return self.t_embedder(t) + self.delta_embedder(r)
        if self.new_condition == "t,t-r":
            return self.t_embedder(t) + self.delta_embedder(t - r)
        if self.new_condition == "r,t-r":
            return self.t_embedder(r) + self.delta_embedder(t - r)
        if self.new_condition == "t,r,t-r":
            return self.t_embedder(t) + self.t_embedder(r) + self.delta_embedder(t - r)
        raise NotImplementedError(f"new_condition={self.new_condition}")

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: [B, N, p*p*C] -> [B, C, H, W]
        b, n, d = x.shape
        p = self.patch_size
        c = self.in_chans
        gh, gw = h // p, w // p
        x = x.reshape(b, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(b, c, h, w)

    def forward(
        self,
        x_t: torch.Tensor,
        t_input: torch.Tensor,
        r_input: torch.Tensor,
        global_attrs: torch.Tensor,
        return_zs: bool = False,
        **_: object,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, h, w = x_t.shape
        assert h == self.img_size and w == self.img_size, "Only fixed-size training is supported in my-app."
        assert c == self.in_chans, "Channel mismatch."

        x = self.x_embedder(x_t) + self.pos_embed  # [B,N,D]
        t_emb = self.t_embedder(t_input).unsqueeze(1)  # [B,1,D]
        delta_emb = self._get_delta_embed(t_input, r_input).unsqueeze(1)  # [B,1,D]
        a_emb = self.attr_embedder(global_attrs).unsqueeze(1)  # [B,1,D]
        cond = t_emb + delta_emb + a_emb

        for blk in self.blocks:
            x = blk(x, cond)

        out_tokens = self.final_layer(x, cond)  # [B,N,p*p*C]
        out = self.unpatchify(out_tokens, h, w)  # [B,C,H,W]

        if return_zs:
            # keep interface compatible; projector loss is not used in my-app
            h_proj = x.mean(dim=1)
            return out, h_proj
        return out, None
