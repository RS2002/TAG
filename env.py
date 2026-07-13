# env.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from utils import EnvironmentReward, evaluate_r


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def compute_local_features(logits, current_ids, is_mask, config):
    probs = F.softmax(logits, dim=-1)
    max_prob, _ = probs.max(dim=-1)

    token_rank_scores = torch.zeros_like(max_prob)
    non_mask = ~is_mask
    B, seq_len = is_mask.shape
    for b in range(B):
        if non_mask[b].any():
            cur_tok = current_ids[b][non_mask[b]]
            probs_b = probs[b, non_mask[b], :]
            sorted_probs, _ = probs_b.sort(dim=-1, descending=True)
            tok_probs = probs_b.gather(1, cur_tok.unsqueeze(-1))
            rank = (sorted_probs == tok_probs).float().argmax(dim=-1) + 1
            token_rank_scores[b, non_mask[b]] = (1.0 / rank.float()).to(token_rank_scores.dtype)

    k = config.top_k_conf
    if k > 0:
        topk_probs, _ = torch.topk(probs, k, dim=-1)
    else:
        topk_probs = torch.empty(B, seq_len, 0, device=logits.device)

    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1, keepdim=True)

    local_feat = torch.cat([
        token_rank_scores.unsqueeze(-1),
        topk_probs,
        entropy
    ], dim=-1).to(logits.dtype)

    return local_feat


def _get_average_attention(model, x, target_layers):
    blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in ('LLaDABlock', 'LLaDASequentialBlock', 'LLaDALlamaBlock'):
            blocks.append(module)
    if not blocks:
        raise RuntimeError("Cannot find LLaDABlock in the model")

    captured = {idx: [] for idx in target_layers}
    original_forwards = {}

    def make_patched_forward(layer_idx):
        target_block = blocks[layer_idx]
        original_attn = target_block.attention
        original_forwards[layer_idx] = original_attn

        def patched_attention(self, q, k, v, attention_bias=None, layer_past=None, use_cache=False):
            att_out, present = original_attn(q, k, v, attention_bias, layer_past, use_cache)
            B, T, C = q.shape
            dtype = k.dtype
            if self.q_norm is not None and self.k_norm is not None:
                q_norm = self.q_norm(q).to(dtype=dtype)
                k_norm = self.k_norm(k).to(dtype=dtype)
            else:
                q_norm, k_norm = q, k
            head_dim = C // self.config.n_heads
            q_reshaped = q_norm.view(B, T, self.config.n_heads, head_dim).transpose(1, 2)
            k_reshaped = k_norm.view(B, T, self.config.effective_n_kv_heads, head_dim).transpose(1, 2)
            if self.config.n_heads != self.config.effective_n_kv_heads:
                k_reshaped = k_reshaped.repeat_interleave(
                    self.config.n_heads // self.config.effective_n_kv_heads, dim=1)
            if self.config.rope:
                q_reshaped, k_reshaped = self.rotary_emb(q_reshaped, k_reshaped)
            query_len, key_len = q_reshaped.shape[-2], k_reshaped.shape[-2]
            if attention_bias is not None:
                attn_bias = attention_bias[:, :, key_len - query_len : key_len, :key_len]
                attn_bias = attn_bias.to(torch.float)
                if attn_bias.dtype != q_reshaped.dtype:
                    attn_bias = attn_bias.to(q_reshaped.dtype)
            else:
                attn_bias = None
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * scale
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias
            attn_probs = F.softmax(attn_weights, dim=-1)
            captured[layer_idx].append(attn_probs.float().detach().cpu().mean(dim=1).squeeze(0))
            return att_out, present
        return patched_attention

    for layer_idx in target_layers:
        blocks[layer_idx].attention = make_patched_forward(layer_idx).__get__(
            blocks[layer_idx], type(blocks[layer_idx]))

    outputs = model(x, attention_mask=None)
    logits = outputs.logits

    for layer_idx in target_layers:
        blocks[layer_idx].attention = original_forwards[layer_idx]

    avg_attn = None
    for layer_idx in target_layers:
        if captured[layer_idx]:
            if avg_attn is None:
                avg_attn = captured[layer_idx][0]
            else:
                avg_attn += captured[layer_idx][0]
    if avg_attn is None:
        raise RuntimeError("No attention captured")
    avg_attn = avg_attn / len(target_layers)
    return logits, avg_attn


def left_pad(tensor, pad_len):
    if pad_len == 0:
        return tensor
    if tensor.dim() == 2:
        pad = torch.zeros(tensor.size(0), pad_len, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([pad, tensor], dim=1)
    elif tensor.dim() == 3:
        pad = torch.zeros(tensor.size(0), pad_len, tensor.size(2), device=tensor.device, dtype=tensor.dtype)
        return torch.cat([pad, tensor], dim=1)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


class LLaDAEnv:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        self.reward_calc = EnvironmentReward(
            max_steps=config.max_mask_steps,
            alpha_speed=config.alpha_speed
        )

        if config.use_hidden_state:
            self.hidden_norm = nn.LayerNorm(config.d_model, elementwise_affine=False).to(self.device)
        else:
            self.hidden_norm = None

        self.prompt = None
        self.ref_answer = None
        self.raw_current_ids = None
        self.raw_is_mask = None
        self.raw_agent_mask = None
        self.raw_seq_len = None
        self.prompt_len = None
        self.gen_len = None
        self.pad_len = None
        self.seq_len = None

        self.last_logits = None
        self.last_hidden = None
        self.local_features = None
        self.global_mean_action = 0.0
        self.step_count = 0
        self.done = False

        self.active_window_start = None
        self.raw_can_unmask = None
        self.raw_can_mask = None
        self.can_unmask = None
        self.can_mask = None
        self.steps_in_current_window = 0

        self.state_history = None

        self.use_attn_features = config.use_attn_features
        self.attn_target_layers = config.attn_target_layers
        self.attn_unmask_ratio = config.attn_unmask_ratio
        self.attn_unmask_min_keep = config.attn_unmask_min_keep
        self.last_attn_map = None

    def _update_window_and_masks(self):
        if not self.config.use_sliding_window:
            raw_can_unmask = torch.ones_like(self.raw_agent_mask, dtype=torch.bool)
            raw_can_mask = torch.ones_like(self.raw_agent_mask, dtype=torch.bool)
            raw_can_mask[~self.raw_agent_mask] = False
        else:
            device = self.device
            agent_mask = self.raw_agent_mask.bool()
            is_mask = self.raw_is_mask
            config = self.config
            gen_start = self.prompt_len
            window_size = config.window_size
            window_stride = config.window_stride
            max_start = gen_start + self.gen_len - window_size
            if max_start < gen_start:
                max_start = gen_start

            prev_start = self.active_window_start
            w_start = prev_start
            while w_start <= max_start:
                w_end = w_start + window_size - 1
                if is_mask[0, w_start:w_end+1].any():
                    break
                w_start += window_stride
            new_start = min(w_start, max_start)
            self.active_window_start = new_start

            active_start = self.active_window_start
            active_end = active_start + window_size - 1

            seq_positions = torch.arange(self.raw_seq_len, device=device)
            raw_can_unmask = torch.zeros_like(agent_mask)
            raw_can_mask = torch.zeros_like(agent_mask)

            in_window = agent_mask & (seq_positions >= active_start) & (seq_positions <= active_end)
            pre_window = agent_mask & (seq_positions < active_start)
            post_window = agent_mask & (seq_positions > active_end)

            raw_can_unmask[in_window] = True
            raw_can_mask[in_window] = True
            raw_can_unmask[pre_window] = True
            raw_can_mask[post_window] = True
            raw_can_unmask[~agent_mask] = True

            if new_start != prev_start:
                self.steps_in_current_window = 0

        self.raw_can_unmask = raw_can_unmask
        self.raw_can_mask = raw_can_mask

        pad_len = self.pad_len
        # raw_can_unmask is (1, raw_len) -> left_pad returns (1, full_len)
        self.can_unmask = left_pad(raw_can_unmask, pad_len)
        self.can_mask = left_pad(raw_can_mask, pad_len)

    def reset(self, prompt, ref_answer, ref_reasoning):
        self.prompt = prompt
        self.ref_answer = ref_answer
        self.reward_calc.reset()

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.device)
        self.prompt_len = prompt_ids.shape[1]
        self.gen_len = self.config.max_gen_length
        self.raw_seq_len = self.prompt_len + self.gen_len
        self.pad_len = self.config.max_total_length - self.raw_seq_len
        if self.pad_len < 0:
            raise ValueError(
                f"max_total_length ({self.config.max_total_length}) < raw_seq_len ({self.raw_seq_len})")
        self.seq_len = self.config.max_total_length

        mask_token_id = self.tokenizer.mask_token_id or 126336
        self.raw_current_ids = torch.full((1, self.raw_seq_len), mask_token_id, device=self.device)
        self.raw_current_ids[:, :self.prompt_len] = prompt_ids

        self.raw_is_mask = torch.ones(1, self.raw_seq_len, dtype=torch.bool, device=self.device)
        self.raw_is_mask[:, :self.prompt_len] = False

        self.raw_agent_mask = torch.zeros(1, self.raw_seq_len, dtype=torch.bool, device=self.device)
        self.raw_agent_mask[:, self.prompt_len:] = True

        self.step_count = 0
        self.done = False
        self.global_mean_action = 0.0
        self.active_window_start = self.prompt_len
        self.raw_can_unmask = None
        self.raw_can_mask = None
        self.can_unmask = None
        self.can_mask = None
        self.steps_in_current_window = 0
        self.state_history = None

        with torch.no_grad():
            if self.use_attn_features:
                self.last_logits, self.last_attn_map = _get_average_attention(
                    self.model, self.raw_current_ids, self.attn_target_layers)
                self.last_hidden = None
            else:
                outputs = self.model(self.raw_current_ids, attention_mask=None,
                                     output_hidden_states=self.config.use_hidden_state)
                self.last_logits = outputs.logits
                self.last_hidden = outputs.hidden_states[-1] if self.config.use_hidden_state else None

        self.local_features = compute_local_features(
            self.last_logits, self.raw_current_ids, self.raw_is_mask, self.config)

        return self._build_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() before stepping again.")

        self.step_count += 1
        self.steps_in_current_window += 1

        raw_action = action[:, self.pad_len:]
        raw_action = raw_action.clone()
        raw_action[~self.raw_agent_mask] = 1

        action_bool = raw_action.bool()
        unmask_positions = action_bool & self.raw_is_mask
        remask_positions = (~action_bool) & (~self.raw_is_mask) & self.raw_agent_mask

        if unmask_positions.any() or remask_positions.any():
            if unmask_positions.any():
                temperature = self.config.generation_temperature
                if temperature > 0:
                    logits_for_unmask = add_gumbel_noise(self.last_logits, temperature)
                else:
                    logits_for_unmask = self.last_logits
                pred_tokens = logits_for_unmask.argmax(dim=-1)
                self.raw_current_ids[unmask_positions] = pred_tokens[unmask_positions]
                self.raw_is_mask[unmask_positions] = False

            if remask_positions.any():
                mask_id = self.tokenizer.mask_token_id or 126336
                self.raw_current_ids[remask_positions] = mask_id
                self.raw_is_mask[remask_positions] = True

            with torch.no_grad():
                if self.use_attn_features:
                    self.last_logits, self.last_attn_map = _get_average_attention(
                        self.model, self.raw_current_ids, self.attn_target_layers)
                    self.last_hidden = None
                else:
                    outputs = self.model(self.raw_current_ids, attention_mask=None,
                                         output_hidden_states=self.config.use_hidden_state)
                    self.last_logits = outputs.logits
                    self.last_hidden = outputs.hidden_states[-1] if self.config.use_hidden_state else None
                self.local_features = compute_local_features(
                    self.last_logits, self.raw_current_ids, self.raw_is_mask, self.config)

        action_float = raw_action.float()
        num_agents = self.raw_agent_mask.sum().float().clamp(min=1)
        self.global_mean_action = (action_float * self.raw_agent_mask.float()).sum().item() / num_agents.item()

        if not (self.raw_is_mask & self.raw_agent_mask).any():
            self.done = True

        reward = self.reward_calc.compute_reward(
            step=self.step_count, done=self.done,
            pred_text=self.get_current_text() if self.done else "",
            ref_answer=self.ref_answer)

        current_correct = float(reward > 0) if self.done else 0.0

        next_state = self._build_state()
        info = {'agent_mask': self.agent_mask, 'current_correct': current_correct}
        return next_state, reward, self.done, info

    def _build_state(self):
        self._update_window_and_masks()

        B, raw_len, device, dtype = 1, self.raw_seq_len, self.device, self.last_logits.dtype
        pad_len = self.pad_len
        full_len = self.seq_len

        # Raw features (unpadded)
        raw_is_pad = torch.zeros(B, raw_len, device=device, dtype=dtype)
        raw_is_prompt = torch.zeros(B, raw_len, device=device, dtype=dtype)
        raw_is_prompt[:, :self.prompt_len] = 1.0
        raw_is_agent = self.raw_agent_mask.float()
        raw_is_mask = self.raw_is_mask.float()
        raw_local = self.local_features

        # Global statistics
        agent_mask_bool = self.raw_agent_mask.bool()
        num_mask = (self.raw_is_mask & agent_mask_bool).sum().float()
        num_agents = agent_mask_bool.sum().clamp(min=1)
        global_mask_count = (num_mask / num_agents).reshape(1, 1, 1)
        global_unmask_count = 1.0 - global_mask_count

        if self.local_features is not None:
            entropy = self.local_features[..., -1]
            agent_entropy = entropy[agent_mask_bool]
            mean_entropy = agent_entropy.mean().reshape(1, 1, 1)
        else:
            mean_entropy = torch.zeros(1, 1, 1, device=device, dtype=dtype)

        mean_action = torch.tensor(self.global_mean_action, device=device, dtype=dtype).view(1, 1, 1)
        step_norm = torch.tensor(self.step_count / self.config.max_mask_steps, device=device, dtype=dtype).view(1, 1, 1)

        probs = F.softmax(self.last_logits, dim=-1)
        confidence, pred_tokens = probs.max(dim=-1)

        raw_can_unmask = self.raw_can_unmask.clone()
        eligible = (raw_can_unmask & self.raw_is_mask.bool()).squeeze(0)

        conf_rank_raw = torch.zeros(B, raw_len, 1, device=device, dtype=dtype)
        if eligible.any():
            conf_vals = confidence[0][eligible]
            sorted_conf, _ = conf_vals.sort(descending=True)
            ranks = (conf_vals.unsqueeze(1) == sorted_conf).float().argmax(dim=1) + 1
            conf_rank_raw[0, eligible, 0] = (ranks / len(sorted_conf)).to(dtype)

        entropy_full = self.local_features[..., -1]
        ent_rank_raw = torch.zeros(B, raw_len, 1, device=device, dtype=dtype)
        if eligible.any():
            ent_vals = entropy_full[0][eligible]
            sorted_ent, _ = ent_vals.sort()
            ranks = (ent_vals.unsqueeze(1) == sorted_ent).float().argmax(dim=1) + 1
            ent_rank_raw[0, eligible, 0] = (ranks / len(sorted_ent)).to(dtype)

        eos_token_id = self.tokenizer.eos_token_id or 128001
        is_eos_pred = (pred_tokens == eos_token_id).float().unsqueeze(-1)
        is_eos_current = (self.raw_current_ids == eos_token_id).float().unsqueeze(-1)

        agent_idx_raw = torch.zeros(B, raw_len, 1, device=device, dtype=dtype)
        if self.raw_agent_mask.any():
            indices = torch.arange(self.gen_len, device=device, dtype=dtype) / self.gen_len
            agent_idx_raw[0, self.raw_agent_mask.squeeze(0), 0] = indices

        attn_score_raw = torch.zeros(B, raw_len, 1, device=device, dtype=dtype)
        attn_rank_raw = torch.zeros(B, raw_len, 1, device=device, dtype=dtype)
        if self.use_attn_features and self.last_attn_map is not None:
            attn_map = self.last_attn_map
            valid_keys = (~self.raw_is_mask).squeeze(0).bool()
            valid_keys_cpu = valid_keys.cpu()
            attn_scores = attn_map[:, valid_keys_cpu].sum(dim=1).to(device=device, dtype=dtype)
            attn_score_raw[0, :, 0] = attn_scores

            if eligible.any():
                scores_in_eligible = attn_scores[eligible]
                sorted_scores, _ = scores_in_eligible.sort(descending=True)
                ranks = (scores_in_eligible.unsqueeze(1) == sorted_scores).float().argmax(dim=1) + 1
                norm_ranks = ranks / len(sorted_scores)
                attn_rank_raw[0, eligible, 0] = norm_ranks.to(dtype)

            if self.attn_unmask_ratio < 1.0:
                cur_eligible = eligible
                if cur_eligible.any():
                    cur_scores = attn_scores[cur_eligible]
                    # k = max(self.attn_unmask_min_keep,
                    #         int(np.ceil(cur_eligible.sum().item() * self.attn_unmask_ratio)))
                    # k = min(cur_eligible.sum().item(), k)
                    k = 16
                    if k < cur_eligible.sum().item():
                        _, topk_idx = torch.topk(cur_scores, k)
                        new_raw_can_unmask = raw_can_unmask.clone()
                        new_raw_can_unmask[0, cur_eligible] = False
                        new_raw_can_unmask[0, cur_eligible.nonzero(as_tuple=True)[0][topk_idx]] = True
                        raw_can_unmask = new_raw_can_unmask
                        self.raw_can_unmask = raw_can_unmask
                        self.can_unmask = left_pad(raw_can_unmask, self.pad_len)
                        eligible = (raw_can_unmask & self.raw_is_mask.bool()).squeeze(0)

        global_stats_raw = torch.cat([
            global_mask_count, global_unmask_count, mean_entropy, mean_action, step_norm
        ], dim=-1)

        if self.config.use_hidden_state and self.last_hidden is not None:
            raw_hidden = self.last_hidden
        else:
            raw_hidden = torch.zeros(B, raw_len, 0, device=device, dtype=dtype)

        def left_pad_feat(feat):
            if feat.dim() == 2:
                return left_pad(feat, pad_len)
            else:
                return left_pad(feat, pad_len)

        is_pad = left_pad_feat(raw_is_pad)
        is_prompt = left_pad_feat(raw_is_prompt)
        is_agent = left_pad_feat(raw_is_agent)
        is_mask = left_pad_feat(raw_is_mask)
        local = left_pad_feat(raw_local)
        conf_rank = left_pad_feat(conf_rank_raw)
        ent_rank = left_pad_feat(ent_rank_raw)
        is_eos_pred_pad = left_pad_feat(is_eos_pred)
        is_eos_current_pad = left_pad_feat(is_eos_current)
        agent_idx = left_pad_feat(agent_idx_raw)
        attn_score = left_pad_feat(attn_score_raw)
        attn_rank = left_pad_feat(attn_rank_raw)
        hidden_pad = left_pad_feat(raw_hidden)

        global_stats = global_stats_raw.expand(1, full_len, -1)
        can_unmask_feat = self.can_unmask.float().unsqueeze(-1)
        can_mask_feat = self.can_mask.float().unsqueeze(-1)

        # Force all components to (1, full_len, D)
        def to_3d(t):
            if t.dim() == 2:
                return t.unsqueeze(-1)
            elif t.dim() == 3:
                return t
            else:
                return t.reshape(1, full_len, -1)

        cur_state = torch.cat([
            to_3d(is_pad),
            to_3d(is_prompt),
            to_3d(is_agent),
            to_3d(is_mask),
            to_3d(local),
            to_3d(global_stats),
            to_3d(can_unmask_feat),
            to_3d(can_mask_feat),
            to_3d(conf_rank),
            to_3d(ent_rank),
            to_3d(is_eos_pred_pad),
            to_3d(is_eos_current_pad),
            to_3d(agent_idx),
            to_3d(attn_score),
            to_3d(attn_rank),
            to_3d(hidden_pad),
        ], dim=-1)

        history_len = self.config.history_len
        if history_len > 1:
            if self.state_history is None:
                self.state_history = deque(maxlen=history_len)
                for _ in range(history_len):
                    self.state_history.append(cur_state.clone())
            else:
                self.state_history.append(cur_state.clone())
            state_list = list(reversed(self.state_history))
            final_state = torch.cat(state_list, dim=-1)
        else:
            final_state = cur_state

        return final_state

    def get_current_text(self):
        return self.tokenizer.decode(self.raw_current_ids[0], skip_special_tokens=True)

    @property
    def agent_mask(self):
        return left_pad(self.raw_agent_mask.float(), self.pad_len).bool()

    @property
    def attention_mask(self):
        raw_attn = torch.ones(1, self.raw_seq_len, device=self.device, dtype=torch.float)
        return left_pad(raw_attn, self.pad_len)

    @property
    def is_mask(self):
        return left_pad(self.raw_is_mask.float(), self.pad_len).bool()

    @property
    def last_logits_padded(self):
        return left_pad(self.last_logits, self.pad_len)