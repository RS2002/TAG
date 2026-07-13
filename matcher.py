# matcher.py
# Bipartite matching action selector under permission constraints.
# Agents with can_unmask=False are forced to MASK (0).
# Agents with can_mask=False are forced to UNMASK (1).
# Agents where both are True form the "free region", where bipartite
# matching is performed with monotonic mask decrease.
# Non-agent positions (e.g., padding, prompt) are forced to MASK (0).

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


class BipartiteMatcher:
    """Selects UNMASK / MASK actions for agent tokens under permission constraints."""

    def __init__(self, config):
        self.k_min = config.k_min
        self.denoising_mode = config.denoising_mode
        self.linear_cap_k = config.linear_cap_k

    def select_action(self, q_values, is_mask, agent_mask, can_unmask, can_mask,
                      epsilon=0.0, step_idx=None):
        """
        Args:
            q_values:    (1, seq_len, 2) with Q for [MASK, UNMASK].
            is_mask:     (1, seq_len) bool, True where token is currently masked.
            agent_mask:  (1, seq_len) bool, True for agent positions.
            can_unmask:  (1, seq_len) bool, True where agent is allowed to UNMASK.
            can_mask:    (1, seq_len) bool, True where agent is allowed to MASK.
            epsilon:     controls Gaussian noise added to Q-values in free region.
            step_idx:    current denoising step index (0‑based), required for linear_cap.
        Returns:
            action: (1, seq_len) long tensor, 1 = UNMASK, 0 = MASK.
        """
        q = q_values.squeeze(0).detach().float().cpu().numpy()
        is_mask_np = is_mask.squeeze(0).cpu().numpy()
        agent_np = agent_mask.squeeze(0).cpu().numpy()
        can_unmask_np = can_unmask.squeeze(0).cpu().numpy()
        can_mask_np = can_mask.squeeze(0).cpu().numpy()

        seq_len = q.shape[0]
        action = np.ones(seq_len, dtype=np.int64)  # default UNMASK

        agent_indices = np.where(agent_np)[0]
        if len(agent_indices) == 0:
            # No agents: force all to MASK
            action[:] = 0
            return torch.tensor(action, device=q_values.device).unsqueeze(0)

        # Force non-agent positions to MASK (padding, prompt, etc.)
        action[~agent_np] = 0

        # Apply hard constraints
        action[agent_np & (~can_unmask_np)] = 0   # force MASK
        action[agent_np & (~can_mask_np)] = 1     # force UNMASK

        # Free region: both unmask and mask allowed
        free_region = agent_np & can_unmask_np & can_mask_np
        free_indices = np.where(free_region)[0]
        n_free = len(free_indices)
        if n_free == 0:
            return torch.tensor(action, device=q_values.device).unsqueeze(0)

        q_free = q[free_indices]               # (n_free, 2)
        m_free = is_mask_np[free_indices].sum()

        # Max mask columns allowed
        if self.denoising_mode == 'linear_cap' and step_idx is not None:
            max_mask_allowed = min(n_free, max(0, n_free - step_idx * self.linear_cap_k))
        else:
            max_mask_allowed = max(0, m_free - max(1, self.k_min))

        # Exploration noise
        q_free_noisy = q_free.copy()
        if epsilon > 0.0:
            noise = np.random.randn(n_free, 2).astype(np.float32) * np.mean(np.abs(q_free_noisy))
            q_free_noisy += noise * epsilon

        # Build cost matrix and solve assignment
        n_unmask_cols = n_free
        n_mask_cols = max_mask_allowed
        cost = np.zeros((n_free, n_unmask_cols + n_mask_cols), dtype=np.float32)
        cost[:, :n_unmask_cols] = q_free_noisy[:, 1:2]   # UNMASK score
        if n_mask_cols > 0:
            cost[:, n_unmask_cols:] = q_free_noisy[:, 0:1]  # MASK score

        row_ind, col_ind = linear_sum_assignment(-cost)
        for i, j in zip(row_ind, col_ind):
            action[free_indices[i]] = 1 if j < n_unmask_cols else 0

        return torch.tensor(action, device=q_values.device).unsqueeze(0)