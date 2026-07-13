# mixer.py
# Mixing networks: QMIX (Transformer‑based) and VDN (simple sum).
# The factory function build_mixer selects the appropriate mixer type.
# All input tensors are explicitly cast to the mixer's internal dtype (bfloat16).
# Padding positions (is_pad=1) are masked in the QMIX Transformer self-attention.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# QMIX Mixer
# ----------------------------------------------------------------------
class QMixer(nn.Module):
    """QMIX mixer with Transformer hypernetwork and padding masking."""

    def __init__(self, n_agents, feat_dim, config):
        super().__init__()
        dtype = config.model_dtype
        self.n_agents = n_agents
        self.embed_dim = config.mixer_embed_dim

        # ---- dimensions of the non‑hidden features ----
        prefix_mask_dim = 4
        local_feat_dim = 1 + config.top_k_conf + 1
        global_feat_dim = config.global_feat_dim
        other_dim = prefix_mask_dim + local_feat_dim + global_feat_dim

        self.use_hidden = config.use_hidden_state
        hidden_proj_dim = config.hidden_proj_dim if self.use_hidden else 0
        self.transformer_dim = hidden_proj_dim + config.other_proj_dim

        if self.use_hidden:
            self.other_proj_dim = self.transformer_dim - hidden_proj_dim
        else:
            self.other_proj_dim = self.transformer_dim

        # projection layers
        self.other_proj = nn.Linear(other_dim, self.other_proj_dim, dtype=dtype)
        if self.use_hidden:
            self.hidden_proj = nn.Linear(config.d_model, hidden_proj_dim, dtype=dtype)

        self.pre_transformer_norm = nn.LayerNorm(self.transformer_dim, dtype=dtype)

        max_len = config.max_total_length
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, self.transformer_dim, dtype=dtype)
        )

        self.transformer_heads = config.mixer_transformer_heads
        self.transformer_layers = config.mixer_transformer_layers
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

        self.agent_weight_head = nn.Linear(self.transformer_dim, self.embed_dim, dtype=dtype)
        self.head_b1 = nn.Linear(self.transformer_dim, self.embed_dim, dtype=dtype)
        self.head_w2 = nn.Linear(self.transformer_dim, self.embed_dim, dtype=dtype)
        self.head_b2 = nn.Linear(self.transformer_dim, 1, dtype=dtype)

        def _init_heads(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.agent_weight_head.apply(_init_heads)
        self.head_b1.apply(_init_heads)
        self.head_w2.apply(_init_heads)
        self.head_b2.apply(_init_heads)

    def forward(self, q_values, agent_features, attention_mask, agent_mask, q_values_2=None):
        """
        Args:
            q_values:        (B, seq_len) advantage values (already mean over agents? Actually per-agent)
            agent_features:  (B, seq_len, state_dim) state tensor
            attention_mask:  (B, seq_len) float, 1 for valid tokens (unused, we use is_pad from agent_features)
            agent_mask:      (B, seq_len) bool, True for agent positions
            q_values_2:      (B, seq_len) optional second Q-values for aux loss
        Returns:
            q_tot: (B, 1) total Q value
            aux_loss: scalar (0 if q_values_2 is None)
        """
        B, seq_len = agent_mask.shape
        target_dtype = self.other_proj.weight.dtype

        # Cast inputs
        q_values = q_values.to(target_dtype)
        agent_features = agent_features.to(target_dtype)
        agent_mask = agent_mask.to(target_dtype)
        if q_values_2 is not None:
            q_values_2 = q_values_2.to(target_dtype)

        # Extract is_pad from the first feature of agent_features (index 0)
        is_pad = agent_features[..., 0].bool()  # (B, seq_len), True for padding

        # ---- Project features ----
        other_dim = self.other_proj.in_features
        other_feat = agent_features[..., :other_dim]
        if self.use_hidden:
            hidden_feat = agent_features[..., other_dim:]
        else:
            hidden_feat = None

        other_out = self.other_proj(other_feat)
        if self.use_hidden:
            hidden_out = self.hidden_proj(hidden_feat)
            x = torch.cat([other_out, hidden_out], dim=-1)
        else:
            x = other_out

        x = self.pre_transformer_norm(x)
        pos = self.pos_embed[:, :seq_len, :]
        x = x + pos

        # Transformer with padding mask
        src_key_padding_mask = is_pad   # (B, seq_len)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Compute agent weights
        raw_weights = torch.abs(self.agent_weight_head(x))
        agent_weights = torch.zeros(B, self.n_agents, self.embed_dim, device=x.device, dtype=x.dtype)
        agent_mask_bool = agent_mask.bool()
        for b in range(B):
            positions = agent_mask_bool[b].nonzero(as_tuple=True)[0]
            num = min(positions.shape[0], self.n_agents)
            if num > 0:
                agent_weights[b, :num] = raw_weights[b, positions[:num]]
        w1 = agent_weights.view(B, self.n_agents * self.embed_dim)

        # Pool over non-padding tokens for global features
        valid_mask = (~is_pad).unsqueeze(-1).to(target_dtype)   # (B, seq_len, 1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B, transformer_dim)

        b1 = self.head_b1(pooled).view(B, 1, self.embed_dim)
        w2 = torch.abs(self.head_w2(pooled)).view(B, self.embed_dim, 1)
        b2 = self.head_b2(pooled).view(B, 1, 1)

        def extract_and_pad(q):
            out = torch.zeros(B, self.n_agents, device=q.device, dtype=q.dtype)
            for b in range(B):
                positions = agent_mask_bool[b].nonzero(as_tuple=True)[0]
                num = min(positions.shape[0], self.n_agents)
                if num > 0:
                    out[b, :num] = q[b, positions[:num]]
            return out

        q_agents = extract_and_pad(q_values)
        if q_values_2 is not None:
            q_agents_2 = extract_and_pad(q_values_2)

        def compute_q_tot(q, w1_tensor):
            q = q.to(w1_tensor.dtype)
            w1_mat = torch.abs(w1_tensor).view(B, self.n_agents, self.embed_dim)
            hidden = F.elu(torch.bmm(q.unsqueeze(1), w1_mat) + b1)
            return (torch.bmm(hidden, w2) + b2).squeeze(-1)

        q_tot = compute_q_tot(q_agents, w1)

        if q_values_2 is not None:
            q_agents_detach = q_agents.detach()
            q_agents_2_detach = q_agents_2.detach()
            q_tot_detach = compute_q_tot(q_agents_detach, w1)
            q_tot2_detach = compute_q_tot(q_agents_2_detach, w1)
            sum_diff = (q_agents_2_detach - q_agents_detach).mean(dim=1, keepdim=True)
            q_diff = q_tot2_detach - q_tot_detach
            aux_loss = torch.relu(-sum_diff * q_diff).mean()
            return q_tot, aux_loss

        return q_tot


# ----------------------------------------------------------------------
# VDN Mixer
# ----------------------------------------------------------------------
class VDNMixer(nn.Module):
    """VDN mixer: simple average of agent advantages."""

    def __init__(self, n_agents, feat_dim, config):
        super().__init__()
        self.n_agents = n_agents

    def forward(self, q_values, agent_features, attention_mask, agent_mask, q_values_2=None):
        """
        Args:
            q_values:    (B, seq_len) advantage values.
            agent_mask:  (B, seq_len) bool, True for agent positions.
        Returns:
            q_tot: (B, 1) averaged advantage.
            aux_loss: 0.0 tensor (for interface compatibility).
        """
        B = q_values.shape[0]
        agent_float = agent_mask.float()
        q_agents = q_values * agent_float
        q_tot = q_agents.sum(dim=1, keepdim=True) / agent_float.sum(dim=1, keepdim=True).clamp(min=1)

        if q_values_2 is not None:
            return q_tot, torch.tensor(0.0, device=q_values.device)
        return q_tot


# ----------------------------------------------------------------------
# Factory function
# ----------------------------------------------------------------------
def build_mixer(n_agents, feat_dim, config):
    mixer_type = config.mixer_type
    if mixer_type.lower() == 'vdn':
        return VDNMixer(n_agents, feat_dim, config)
    else:
        return QMixer(n_agents, feat_dim, config)