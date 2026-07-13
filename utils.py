# utils.py
# Utility functions for model loading, data processing, answer extraction,
# environment reward computation, and the RewardModel for potential‑based shaping.

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from fractions import Fraction


def load_llada_model_and_tokenizer(config):
    """
    Load the frozen LLaDA backbone model and tokenizer from local path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=config.model_dtype,
        device_map=config.device_map,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.eval()

    if tokenizer.mask_token_id is None:
        tokenizer.mask_token = '[MASK]'
        tokenizer.mask_token_id = getattr(model.config, 'mask_token_id', 126336)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.pad_token_id = getattr(model.config, 'pad_token_id', 126081)
    return model, tokenizer


def load_gsm8k_data(data_path, file_name):
    """
    Load GSM8K dataset from jsonl file.
    Returns:
        problems: list of question strings
        references: list of dicts with keys 'reasoning' and 'answer'
    """
    with open(f"{data_path}/{file_name}", "r", encoding="utf-8") as f:
        lines = f.readlines()
    problems = []
    references = []
    for line in lines:
        item = json.loads(line)
        question = item["question"]
        answer_raw = item["answer"]
        parts = answer_raw.split("####")
        if len(parts) >= 2:
            reasoning = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            reasoning = answer_raw
            final_answer = answer_raw
        problems.append(question)
        references.append({"reasoning": reasoning, "answer": final_answer})
    return problems, references


def extract_gsm8k_answer(text):
    """
    Extract final answer from generated text for GSM8K.
    Priority: \boxed{...} > #### > last number.
    Returns the extracted answer string or '[invalid]' if extraction fails.
    """
    if not text:
        return "[invalid]"
    
    # 1. Try \boxed{} pattern (allow one level of nested braces for fractions etc.)
    # boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    # if boxed_match:
    #     return boxed_match.group(1).strip()
    
    # 2. Try #### pattern
    matches = re.findall(r"####\s*(\-?[0-9\.\,]+)", text)
    if matches:
        return matches[-1].strip().replace(",", "")
    
    # 3. Last resort: last number in text
    numbers = re.findall(r"[\-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return "[invalid]"


def normalize_answer(ans):
    """
    Normalize an answer string for numerical comparison.
    Supports integers, floats, and fractions.
    """
    if ans is None:
        return "[invalid]"
    if ans == "[invalid]":
        return ans

    cleaned = ans.replace(",", "").replace(" ", "")
    if not cleaned:
        return "[invalid]"

    if '/' in cleaned:
        try:
            return Fraction(cleaned)
        except (ValueError, ZeroDivisionError):
            pass

    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def is_answer_correct(pred_text, ref_answer):
    """
    Check if the predicted answer matches the reference answer.
    Performs numerical equivalence check with tolerance for floating point,
    and exact comparison for fractions.
    """
    pred_ans = extract_gsm8k_answer(pred_text)
    pred_norm = normalize_answer(pred_ans)
    ref_norm = normalize_answer(ref_answer)

    if pred_norm == "[invalid]" or ref_norm == "[invalid]":
        return False

    if isinstance(pred_norm, Fraction) and isinstance(ref_norm, Fraction):
        return pred_norm == ref_norm

    if isinstance(pred_norm, Fraction) and isinstance(ref_norm, float):
        return abs(float(pred_norm) - ref_norm) < 1e-5
    if isinstance(pred_norm, float) and isinstance(ref_norm, Fraction):
        return abs(pred_norm - float(ref_norm)) < 1e-5

    if isinstance(pred_norm, float) and isinstance(ref_norm, float):
        return abs(pred_norm - ref_norm) < 1e-5

    return str(pred_norm) == str(ref_norm)


def evaluate_r(text, ref_answer):
    """
    Compute the binary verifiable reward r(x) for a given generation.
    Returns 1.0 if the extracted answer is correct, else 0.0.
    """
    return 1.0 if is_answer_correct(text, ref_answer) else 0.0


class EnvironmentReward:
    """
    Computes the task‑related environment reward.
    Only includes the final outcome reward (speed‑dependent) when the episode is done
    and the answer is correct. All dense signals are moved to the potential function.
    """
    def __init__(self, max_steps, alpha_speed):
        self.max_steps = max_steps
        self.alpha_speed = alpha_speed

    def reset(self):
        pass

    def compute_reward(self, step, done, pred_text, ref_answer):
        """
        Args:
            step: current step number (int).
            done: bool, whether the episode has finished.
            pred_text: generated text (only used when done).
            ref_answer: reference answer string.
        Returns:
            reward (float): (1 - step/max_steps)^alpha_speed if done and correct, else 0.0.
        """
        if not done:
            return 0.0
        if evaluate_r(pred_text, ref_answer) == 1.0:
            # return (1.0 - (step-1) / self.max_steps) ** self.alpha_speed
            return 1.0
        else:
            return 0.0


def get_heuristic_action(env, config):
    """
    Select actions based on confidence‑based heuristic (high‑confidence first).
    Uses the raw environment attributes (unpadded) to compute action on raw sequence,
    then left-pads to full length for compatibility with Q-network output.

    Args:
        env: LLaDAEnv instance (must have raw attributes: raw_is_mask, raw_agent_mask,
             raw_can_unmask, raw_can_mask, last_logits, raw_seq_len, pad_len).
        config: Config object with curriculum parameters.

    Returns:
        action: (1, seq_len) long tensor with 1=UNMASK, 0=MASK (padded).
    """
    device = env.device
    raw_seq_len = env.raw_seq_len
    pad_len = env.pad_len

    # Raw tensors (unpadded)
    raw_is_mask = env.raw_is_mask                     # (1, raw_seq_len) bool
    raw_agent_mask = env.raw_agent_mask               # (1, raw_seq_len) bool
    raw_can_unmask = env.raw_can_unmask               # (1, raw_seq_len) bool
    raw_can_mask = env.raw_can_mask                   # (1, raw_seq_len) bool
    logits = env.last_logits                          # (1, raw_seq_len, vocab_size)

    raw_action = torch.ones(1, raw_seq_len, dtype=torch.long, device=device)

    # Force non-agent positions to MASK (0)
    raw_action[~raw_agent_mask] = 0

    # Candidates: positions that are agent and currently masked
    candidate_mask = (raw_is_mask & raw_agent_mask).squeeze(0)   # (raw_seq_len,)
    if candidate_mask.sum() == 0:
        # No masked agents: return padded action (all MASK)
        return F.pad(raw_action, (pad_len, 0), value=0)

    # Enforce hard constraints from raw_can_unmask / raw_can_mask
    force_mask = (~raw_can_unmask) & raw_agent_mask
    raw_action[force_mask] = 0
    force_unmask = (~raw_can_mask) & raw_agent_mask
    raw_action[force_unmask] = 1

    # Free region: both unmask and mask allowed
    free_region = raw_can_unmask & raw_can_mask & raw_agent_mask   # (1, raw_seq_len)
    free_region_sq = free_region.squeeze(0)                         # (raw_seq_len,)
    free_candidate = candidate_mask & free_region_sq
    num_free_candidates = free_candidate.sum().item()

    if num_free_candidates == 0:
        # No free candidates: return padded action
        return F.pad(raw_action, (pad_len, 0), value=0)

    # Determine how many tokens to unmask this step
    n_unmask = np.random.randint(config.curriculum_min_unmask,
                                 config.curriculum_max_unmask + 1)
    n_unmask = min(n_unmask, num_free_candidates)

    # If unmasking all free candidates, action already has 1 for them (default)
    if n_unmask >= num_free_candidates:
        return F.pad(raw_action, (pad_len, 0), value=0)

    # Compute confidence (max softmax probability) for all raw positions
    probs = F.softmax(logits, dim=-1)          # (1, raw_seq_len, vocab)
    max_conf, _ = probs.max(dim=-1)            # (1, raw_seq_len)
    max_conf = max_conf[0]                     # (raw_seq_len,)

    # Select low‑confidence tokens in the free region to keep masked
    conf_free = max_conf[free_candidate]       # (num_free_candidates,)
    num_keep = num_free_candidates - n_unmask
    _, low_indices = torch.topk(conf_free, num_keep, largest=False)

    free_positions = free_candidate.nonzero(as_tuple=True)[0]
    keep_positions = free_positions[low_indices]
    raw_action[0, keep_positions] = 0          # MASK low‑confidence tokens

    # Left-pad to full sequence length
    action = F.pad(raw_action, (pad_len, 0), value=0)
    return action


class RewardModel(nn.Module):
    """
    Reward model that estimates the probability of eventual success
    given the current state.
    Architecture: separate projection for non‑hidden and hidden features,
    Transformer encoder, pooling over valid (non-pad) tokens, and a classification head.
    This model is used to construct the potential function for PBRS.
    Padding positions are masked in self-attention and excluded from pooling.
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

        self.hidden_proj_dim = config.hidden_proj_dim if self.use_hidden else 0
        self.transformer_dim = self.hidden_proj_dim + config.other_proj_dim
        self.other_proj_dim = self.transformer_dim - self.hidden_proj_dim if self.use_hidden else self.transformer_dim

        # ---- projection layers ----
        self.other_proj = nn.Linear(other_dim * config.history_len, self.other_proj_dim, dtype=dtype)
        if self.use_hidden:
            self.hidden_proj = nn.Linear(config.d_model * config.history_len, self.hidden_proj_dim, dtype=dtype)

        # LayerNorm after concatenation
        self.pre_transformer_norm = nn.LayerNorm(self.transformer_dim, dtype=dtype)

        # Positional embedding
        max_len = config.max_total_length
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, self.transformer_dim, dtype=dtype)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=config.rm_heads,
            dim_feedforward=self.transformer_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        ).to(dtype)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.rm_layers)

        # Output head
        self.out_proj = nn.Linear(self.transformer_dim, 1, dtype=dtype)

    def forward(self, state_tensor, attention_mask):
        """
        Args:
            state_tensor:  (B, seq_len, state_dim)
            attention_mask: (B, seq_len) float, 1 for valid tokens, 0 for PAD.
        Returns:
            prob: (B, 1) probability of eventual correct answer.
        """
        B, seq_len, _ = state_tensor.shape
        target_dtype = self.other_proj.weight.dtype
        state = state_tensor.to(dtype=target_dtype)

        # Extract is_pad from the first feature (index 0)
        is_pad = state[..., 0].bool()   # (B, seq_len), True for padding positions

        # ---- slice the state tensor ----
        other_dim = self.other_proj.in_features
        if self.use_hidden:
            other_feat = state[..., :other_dim]
            hidden_feat = state[..., other_dim:]
        else:
            other_feat = state
            hidden_feat = None

        # ---- project separately ----
        other_out = self.other_proj(other_feat)                     # (B, seq_len, other_proj_dim)
        if self.use_hidden:
            hidden_out = self.hidden_proj(hidden_feat)               # (B, seq_len, hidden_proj_dim)
            x_ori = torch.cat([other_out, hidden_out], dim=-1)
        else:
            x_ori = other_out

        # ---- normalize + position embedding ----
        x = self.pre_transformer_norm(x_ori)
        pos = self.pos_embed[:, :seq_len, :]
        x = x + pos

        # ---- Transformer encoder with padding mask ----
        src_key_padding_mask = is_pad   # (B, seq_len), True for positions to ignore
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, seq_len, transformer_dim)

        # ---- Pooling over non-padding tokens ----
        valid_mask = (~is_pad).unsqueeze(-1).to(target_dtype)   # (B, seq_len, 1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B, transformer_dim)

        # ---- Output probability ----
        logit = self.out_proj(pooled)            # (B, 1)
        prob = torch.sigmoid(logit)              # (B, 1)
        return prob