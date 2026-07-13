# replay_buffer.py
# Prioritized experience replay buffer for QMIX training.
# Stores unified state tensors instead of separate components.
# No longer stores global states; the mixer dynamically reads features from the state tensors.

import numpy as np
import torch

class PrioritizedReplayBuffer:
    """
    Stores transitions and samples them with probability proportional to priority.
    Priority is based on TD-error magnitude.
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum number of transitions to store.
            alpha:    Exponent that determines how much prioritization is used (0 = uniform).
            beta:     Initial value of beta for importance-sampling correction.
            beta_increment: Amount to increase beta per sampling call.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Storage for transition components
        self.state_tensors = []          # (1, seq_len, state_dim) tensors
        self.actions = []                 # (1, seq_len) tensors
        self.rewards = []                 # float scalars
        self.next_state_tensors = []     # (1, seq_len, state_dim) tensors
        self.dones = []                   # bool scalars
        self.agent_masks = []             # (1, seq_len) bool tensors
        self.attention_masks = []         # (1, seq_len) float tensors (0/1)
        self.steps = []                   # int (step index of the current state)

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, agent_mask, attention_mask, step):
        """
        Save a transition.
        state:           tensor (1, seq_len, state_dim)
        action:          tensor (1, seq_len)
        reward:          float
        next_state:      tensor (1, seq_len, state_dim)
        done:            bool
        agent_mask:      tensor (1, seq_len) bool
        attention_mask:  tensor (1, seq_len) float
        step:            int (current step index, 0-based)
        """
        state_cpu = state.cpu()
        action_cpu = action.cpu()
        next_state_cpu = next_state.cpu()
        agent_mask_cpu = agent_mask.cpu()
        attention_mask_cpu = attention_mask.cpu()

        if self.size < self.capacity:
            self.state_tensors.append(state_cpu)
            self.actions.append(action_cpu)
            self.rewards.append(reward)
            self.next_state_tensors.append(next_state_cpu)
            self.dones.append(done)
            self.agent_masks.append(agent_mask_cpu)
            self.attention_masks.append(attention_mask_cpu)
            self.steps.append(step)
            self.size += 1
        else:
            idx = self.pos
            self.state_tensors[idx] = state_cpu
            self.actions[idx] = action_cpu
            self.rewards[idx] = reward
            self.next_state_tensors[idx] = next_state_cpu
            self.dones[idx] = done
            self.agent_masks[idx] = agent_mask_cpu
            self.attention_masks[idx] = attention_mask_cpu
            self.steps[idx] = step

        # Assign maximum priority to new transition
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns a tuple containing:
            state_tensors_batch, actions_batch, rewards_batch, next_state_tensors_batch,
            dones_batch, agent_masks_batch, attention_masks_batch, steps_batch, indices, weights
        """
        if self.size == 0:
            raise ValueError("Buffer is empty")

        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sum_priorities = scaled_priorities.sum()
        if sum_priorities <= 0.0:
            probs = np.ones(self.size) / self.size
        else:
            probs = scaled_priorities / sum_priorities

        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)

        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Gather batch elements
        state_batch = [self.state_tensors[i] for i in indices]
        action_batch = [self.actions[i] for i in indices]
        reward_batch = [self.rewards[i] for i in indices]
        next_state_batch = [self.next_state_tensors[i] for i in indices]
        done_batch = [self.dones[i] for i in indices]
        agent_mask_batch = [self.agent_masks[i] for i in indices]
        attn_mask_batch = [self.attention_masks[i] for i in indices]
        step_batch = [self.steps[i] for i in indices]

        # Stack tensors (along batch dimension)
        states = torch.cat(state_batch, dim=0)          # (B, seq_len, state_dim)
        actions = torch.cat(action_batch, dim=0)        # (B, seq_len)
        next_states = torch.cat(next_state_batch, dim=0)
        agent_masks = torch.cat(agent_mask_batch, dim=0)
        attn_masks = torch.cat(attn_mask_batch, dim=0)

        rewards = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32)
        steps_tensor = torch.tensor(step_batch, dtype=torch.long)  # (B,)

        return (states, actions, rewards, next_states, dones, agent_masks, attn_masks,
                steps_tensor, indices, weights)

    def update_priorities(self, indices, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return self.size