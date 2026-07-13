# inference.py
# Inference script for TAG model on GSM8K test set with optional initial confidence guidance.
# Supports per‑window or global guidance.
# Saves detailed results to a JSON file and shows live progress with cumulative metrics.

import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from utils import (load_llada_model_and_tokenizer, load_gsm8k_data,
                   is_answer_correct, get_heuristic_action)
from env import LLaDAEnv
from qnet import build_qnet
from matcher import BipartiteMatcher


def evaluate_with_guidance(checkpoint_path, K_infer, per_window, max_samples=None, output_json="inference_results.json"):
    """
    Evaluate the TAG model on GSM8K test set with initial confidence guidance.
    Saves per‑problem results to a JSON file.

    Args:
        checkpoint_path: path to Q‑network checkpoint.
        K_infer: number of initial guidance steps (per window if per_window=True, else global).
        per_window: whether to distribute guidance to each sliding window.
        max_samples: limit number of test problems.
        output_json: file path for saving results.
    """
    config = Config()
    device = torch.device(config.device_map if torch.cuda.is_available() else "cpu")

    # Load backbone and tokenizer
    print("Loading LLaDA backbone and tokenizer...")
    model, tokenizer = load_llada_model_and_tokenizer(config)

    # Load test data
    print("Loading GSM8K test data...")
    problems, references = load_gsm8k_data(config.data_path, config.test_file)

    if max_samples is not None and max_samples < len(problems):
        indices = random.sample(range(len(problems)), max_samples)
        problems = [problems[i] for i in indices]
        references = [references[i] for i in indices]
        print(f"Evaluating on {max_samples} random samples.")

    # Create environment
    env = LLaDAEnv(config, model, tokenizer)

    # Infer state_dim
    state_sample = env.reset(problems[0], references[0]["answer"], references[0]["reasoning"])
    config.state_dim = state_sample.shape[-1]

    # Load Q‑network
    qnet = build_qnet(config).to(device)
    if checkpoint_path is not None:
        print(f"Loading Q‑network checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        qnet.load_state_dict(state_dict)
    else:
        print("No checkpoint provided, using randomly initialized Q‑network (results will be poor).")
    qnet.eval()

    matcher = BipartiteMatcher(config)

    # Pre‑compute number of windows if per_window mode is used
    if config.use_sliding_window and per_window:
        num_windows = int(np.ceil(config.max_gen_length / config.window_stride))
    else:
        num_windows = 1

    correct = 0
    total_steps = 0
    total_reward = 0.0
    num_problems = len(problems)
    results = []  # to store per‑problem results

    pbar = tqdm(range(num_problems), desc="Inference")
    for idx in pbar:
        prompt = problems[idx]
        ref_answer = references[idx]["answer"]

        state_tensor = env.reset(prompt, ref_answer, "")
        agent_mask = env.agent_mask.to(device)

        done = False
        steps = 0
        episode_reward = 0.0
        with torch.no_grad():
            while not done:
                # Determine if we should use heuristic guidance
                use_heuristic = False
                if K_infer > 0:
                    if per_window and config.use_sliding_window:
                        # Per‑window guidance: use heuristic for first K_infer steps in each window
                        use_heuristic = (env.steps_in_current_window < K_infer)
                    else:
                        # Global guidance: use heuristic for first K_infer steps of the episode
                        use_heuristic = (steps < K_infer)

                if use_heuristic:
                    action = get_heuristic_action(env, config)
                else:
                    # Use Q‑network (only advantages needed for action selection)
                    q_adv, _ = qnet(state_tensor.to(device), agent_mask)
                    is_mask = env.is_mask.to(device)
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
        is_correct = is_answer_correct(pred_text, ref_answer)
        if is_correct:
            correct += 1

        total_steps += steps
        total_reward += episode_reward

        # Save per‑problem result
        results.append({
            "problem": prompt,
            "prediction": pred_text,
            "correct": is_correct,
            "reference_answer": ref_answer
        })

        # Update progress bar with current and cumulative metrics
        current_acc = correct / (idx + 1) if (idx + 1) > 0 else 0.0
        avg_steps = total_steps / (idx + 1) if (idx + 1) > 0 else 0.0
        avg_reward = total_reward / (idx + 1) if (idx + 1) > 0 else 0.0

        pbar.set_postfix({
            "cur_steps": steps,
            "acc": f"{current_acc:.3f}",
            "avg_steps": f"{avg_steps:.1f}",
            "avg_rew": f"{avg_reward:.3f}"
        })

    final_accuracy = correct / num_problems if num_problems > 0 else 0.0
    final_avg_steps = total_steps / num_problems if num_problems > 0 else 0.0
    final_avg_reward = total_reward / num_problems if num_problems > 0 else 0.0

    print(f"\nInference Results ({num_problems} samples):")
    print(f"  Accuracy: {final_accuracy:.4f} ({correct}/{num_problems})")
    print(f"  Average steps: {final_avg_steps:.2f}")
    print(f"  Average reward: {final_avg_reward:.4f}")

    # Save results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {output_json}")

    return {"accuracy": final_accuracy, "avg_steps": final_avg_steps, "avg_reward": final_avg_reward}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with TAG model + optional guidance")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/qnet_ep1200.pth",
                        help="Path to Q‑network checkpoint (.pth)")
    parser.add_argument("--K_infer", type=int, default=0,
                        help="Number of initial confidence‑guided steps")
    parser.add_argument("--per_window", action="store_true",
                        help="Distribute guidance steps to each sliding window")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Number of test samples to evaluate (default: full test set)")
    parser.add_argument("--output_json", type=str, default="inference_results.json",
                        help="Path for the detailed JSON output file")
    args = parser.parse_args()

    evaluate_with_guidance(args.checkpoint, args.K_infer, args.per_window,
                           args.max_samples, args.output_json)