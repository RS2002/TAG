# eval.py
# Evaluation script for TAG model on GSM8K test set.
# Uses the unified state tensor interface (state_tensor) and the Q‑net API.
# Mixer is not needed for evaluation.

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from utils import load_llada_model_and_tokenizer, load_gsm8k_data, is_answer_correct
from env import LLaDAEnv
from qnet import build_qnet
from matcher import BipartiteMatcher


def evaluate(checkpoint_path=None, model=None, tokenizer=None, max_samples=None):
    """
    Evaluate the TAG model on GSM8K test set.

    Args:
        checkpoint_path (str, optional): Path to Q-network state dict.
        model: Pre-loaded frozen LLaDA model (optional).
        tokenizer: Pre-loaded tokenizer (optional).
        max_samples (int, optional): If set, evaluate only a random subset.

    Returns:
        dict: {"accuracy": float, "avg_steps": float, "avg_reward": float}
    """
    config = Config()
    device = torch.device(config.device_map if torch.cuda.is_available() else "cpu")

    if model is None or tokenizer is None:
        print("Loading LLaDA backbone and tokenizer...")
        model, tokenizer = load_llada_model_and_tokenizer(config)
    else:
        print("Using pre-loaded backbone and tokenizer.")

    print("Loading GSM8K test data...")
    problems, references = load_gsm8k_data(config.data_path, config.test_file)

    if max_samples is not None and max_samples < len(problems):
        indices = random.sample(range(len(problems)), max_samples)
        problems = [problems[i] for i in indices]
        references = [references[i] for i in indices]
        print(f"Fast evaluation on {max_samples} random samples.")

    env = LLaDAEnv(config, model, tokenizer)

    state_sample = env.reset(problems[0], references[0]["answer"], references[0]["reasoning"])
    config.state_dim = state_sample.shape[-1]

    qnet = build_qnet(config).to(device)
    if checkpoint_path is not None:
        print(f"Loading Q-network checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        qnet.load_state_dict(state_dict)
    else:
        print("No checkpoint provided, using randomly initialized Q-network.")
    qnet.eval()
    matcher = BipartiteMatcher(config)

    correct = 0
    total_steps = 0
    total_reward = 0.0
    num_problems = len(problems)

    print(f"Evaluating on {num_problems} problems...")
    for idx in tqdm(range(num_problems), desc="Eval"):
        prompt = problems[idx]
        ref_answer = references[idx]["answer"]

        state_tensor = env.reset(prompt, ref_answer, "")
        agent_mask = env.agent_mask.to(device)

        done = False
        steps = 0
        episode_reward = 0.0
        with torch.no_grad():
            while not done:
                is_mask = env.is_mask.to(device)
                state = state_tensor.to(device)

                # Q-network returns advantages and global value; only advantages are used for action selection
                q_adv, _ = qnet(state, agent_mask)
                can_unmask = env.can_unmask.to(device)
                can_mask = env.can_mask.to(device)

                action = matcher.select_action(q_adv, is_mask, agent_mask,
                                               can_unmask, can_mask, epsilon=0.0,
                                               step_idx=steps)

                next_state_tensor, reward, done, info = env.step(action)
                episode_reward += reward
                state_tensor = next_state_tensor
                steps += 1

                if steps >= config.max_mask_steps:
                    break

        pred_text = env.get_current_text()
        if is_answer_correct(pred_text, ref_answer):
            correct += 1
        total_steps += steps
        total_reward += episode_reward

    accuracy = correct / num_problems if num_problems > 0 else 0.0
    avg_steps = total_steps / num_problems if num_problems > 0 else 0.0
    avg_reward = total_reward / num_problems if num_problems > 0 else 0.0

    print(f"\nEvaluation Results ({num_problems} samples):")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{num_problems})")
    print(f"  Average steps: {avg_steps:.2f}")
    print(f"  Average reward: {avg_reward:.4f}")

    return {"accuracy": accuracy, "avg_steps": avg_steps, "avg_reward": avg_reward}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TAG on GSM8K test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Q-network checkpoint (.pth)")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of test samples to evaluate (default: full test set)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (overrides config.device_map)")
    args = parser.parse_args()

    if args.device is not None:
        Config.device_map = args.device

    evaluate(checkpoint_path=args.checkpoint, max_samples=args.max_samples)