# qnet.py
# Q-network for multi-agent discrete diffusion unmasking.
# Transformer-only version that receives a unified state tensor.
# The hidden state portion is projected separately from the other features,
# and a LayerNorm is applied before the Transformer to balance the two streams.
# Padding positions (is_pad=1) are masked in the self-attention.

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetTransformer(nn.Module):
    """
    Transformer‑based Q‑network.
    Input:
        state_tensor  : (B, seq_len, state_dim) – pre‑computed per‑token features.
                        The trailing segment of dimension d_model contains the
                        (optionally normalised) LLaDA hidden states.
        agent_mask    : (B, seq_len) bool, True for agent positions;
                        PAD positions can be masked from self‑attention.
    Output:
        q_adv    : (B, seq_len, 2)  advantages for [MASK, UNMASK]
        v_global : (B, 1)           global state value V(s)
    """
    def __init__(self, config):
        super().__init__()
        dtype = config.model_dtype
        self.use_hidden = config.use_hidden_state

        # ---- dimensions of the non‑hidden features ----
        prefix_mask_dim = 4                      # is_pad, is_prompt, is_agent, is_mask
        local_feat_dim = 1 + config.top_k_conf + 1   # rank_score + topk + entropy
        global_feat_dim = config.global_feat_dim      # mask_count, unmask_count, mean_entropy, mean_action, step_norm
        other_dim = prefix_mask_dim + local_feat_dim + global_feat_dim

        self.other_proj_dim = config.other_proj_dim              # projection dim for non‑hidden features
        self.hidden_proj_dim = config.hidden_proj_dim if self.use_hidden else 0

        # total transformer dimension after separate projections
        self.transformer_dim = self.other_proj_dim + self.hidden_proj_dim

        # ---- projection layers ----
        self.other_proj = nn.Linear(other_dim * config.history_len, self.other_proj_dim, dtype=dtype)
        if self.use_hidden:
            self.hidden_proj = nn.Linear(config.d_model * config.history_len, self.hidden_proj_dim, dtype=dtype)

        # LayerNorm after concatenation to balance the two streams
        self.pre_transformer_norm = nn.LayerNorm(self.transformer_dim, dtype=dtype)

        # Positional embedding (fixed sequence length)
        max_len = config.max_total_length
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, self.transformer_dim, dtype=dtype)
        )

        # Transformer encoder
        self.transformer_heads = config.transformer_heads
        self.transformer_layers = config.transformer_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        ).to(dtype)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)

        # Output head for advantages (2 actions)
        self.out_proj = nn.Linear(self.transformer_dim, 2, dtype=dtype)

        # Head for global state value V(s)
        self.v_head = nn.Sequential(
            nn.Linear(self.transformer_dim, self.transformer_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(self.transformer_dim, 1, dtype=dtype)
        )

        # # Bounded advantage scaling
        # self.beta = nn.Parameter(torch.randn([1], dtype=dtype))
        # self.tanh = nn.Tanh()
        # self.softplus = nn.Softplus()
        self.duel = config.use_dueling_mixer


        # Conservative initialization for output heads
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-2)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        for layer in self.v_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1e-2)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state_tensor, agent_mask):
        """
        Args:
            state_tensor: (B, seq_len, state_dim) float tensor.
            agent_mask:   (B, seq_len) bool (unused for masking, kept for compatibility).
        Returns:
            q_adv    : (B, seq_len, 2)
            v_global : (B, 1)
        """
        B, seq_len, _ = state_tensor.shape
        target_dtype = self.other_proj.weight.dtype
        state = state_tensor.to(dtype=target_dtype)

        # Extract is_pad from the first feature (index 0)
        is_pad = state[..., 0].bool()   # (B, seq_len), True for padding positions

        # ---- slice the state ----
        other_dim = self.other_proj.in_features
        other_feat = state[..., :other_dim]      # (B, seq_len, other_dim)
        if self.use_hidden:
            hidden_feat = state[..., other_dim:]  # (B, seq_len, hidden_dim)
        else:
            hidden_feat = None

        # ---- project separately ----
        other_out = self.other_proj(other_feat)                     # (B, seq_len, other_proj_dim)
        if self.use_hidden:
            hidden_out = self.hidden_proj(hidden_feat)               # (B, seq_len, hidden_proj_dim)
            x_ori = torch.cat([other_out, hidden_out], dim=-1)      # (B, seq_len, transformer_dim)
        else:
            x_ori = other_out

        # ---- normalize before adding positional embedding ----
        x = self.pre_transformer_norm(x_ori)

        # ---- positional embedding ----
        pos = self.pos_embed[:, :seq_len, :]
        x = x + pos

        # ---- Transformer encoder with padding mask ----
        # src_key_padding_mask: True for positions to ignore (padding)
        src_key_padding_mask = is_pad   # (B, seq_len)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # ---- Advantage output with bounded range ----
        q_adv = self.out_proj(x)                     # (B, seq_len, 2)
        # q_adv = self.tanh(q_adv) * self.softplus(self.beta)   # bounded advantages

        # ---- Global state value V(s) via simple mean pooling over all tokens ----
        # Use only non-padding positions for pooling
        if self.duel:
            valid_mask = (~is_pad).unsqueeze(-1).to(target_dtype)   # (B, seq_len, 1)
            pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B, transformer_dim)
            v_global = self.v_head(pooled)                # (B, 1)
        else:
            v_global = 0

        return q_adv, v_global


def build_qnet(config):
    """Factory function that returns a QNetTransformer instance."""
    return QNetTransformer(config)