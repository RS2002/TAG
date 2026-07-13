# train.py
# Training script for TAG with QMIX/VDN, frozen Reward Model, and potential‑based shaping.
# The environment provides only the final task reward.
# Dense process rewards are computed externally via PBRS (RM) and optional entropy gain,
# weighted by coefficients from config.
# Supports loading pre‑trained Q‑net, mixer, and RM; RM training can be enabled/disabled.
# Each problem can be repeated several times (config.episodes_per_problem) before switching.
#
# Curriculum learning: early episodes start with confidence‑based warmup steps
# before the Q‑net takes over. The maximum number of warmup steps K_max decays over time.
# When curriculum_per_window_mode is True, warmup is applied per sliding window:
#   each window gets up to K_per_window heuristic steps, where
#   K_per_window = max(1, K_max // num_windows).
# Otherwise, warmup is global (first K steps of the episode).

import os
import sys
import csv
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import (load_llada_model_and_tokenizer, load_gsm8k_data, RewardModel,
                   EnvironmentReward, get_heuristic_action)
from env import LLaDAEnv
from qnet import build_qnet
from matcher import BipartiteMatcher
from replay_buffer import PrioritizedReplayBuffer
from mixer import build_mixer
from eval import evaluate


def focal_loss(pred, target, alpha=0.25, gamma=2.0, pos_weight=None):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * ((1 - p_t) ** gamma)
    loss = focal_weight * bce
    if pos_weight is not None:
        loss = loss * (pos_weight * target + (1 - target))
    return loss.mean()


def train():
    config = Config()
    device = torch.device(config.device_map if torch.cuda.is_available() else "cpu")
    ema_pos_ratio = 0.5

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("Loading LLaDA backbone and tokenizer...")
    model, tokenizer = load_llada_model_and_tokenizer(config)

    print("Loading GSM8K training data...")
    problems, references = load_gsm8k_data(config.data_path, config.train_file)
    num_train = len(problems)
    print(f"Training samples: {num_train}")

    env = LLaDAEnv(config, model, tokenizer)

    state_sample = env.reset(problems[0], references[0]["answer"], references[0]["reasoning"])
    config.state_dim = state_sample.shape[-1]
    print(f"State dimension: {config.state_dim}")

    # Feature indices in the state tensor (based on order in env._build_state)
    prefix_mask_dim = 4
    local_feat_dim = 1 + config.top_k_conf + 1
    glob_start = prefix_mask_dim + local_feat_dim
    # can_unmask and can_mask are at positions glob_start + 5 and glob_start + 6
    can_unmask_idx = glob_start + 5
    can_mask_idx = glob_start + 6
    # entropy is at the last position of local features
    entropy_idx = prefix_mask_dim + local_feat_dim - 1
    # confidence is the first element of local features (after token_rank_score)
    # local features: [token_rank_score, topk_probs..., entropy]
    confidence_idx = prefix_mask_dim   # token_rank_score position (index 4)
    mask_idx = prefix_mask_dim - 1     # is_mask is at index 3

    if config.curriculum_per_window_mode:
        num_windows = int(np.ceil(config.max_gen_length / config.window_stride))
    else:
        num_windows = 1

    # --- Q-network ---
    qnet = build_qnet(config).to(device)
    if config.use_pretrained_qnet:
        print("Loading pre‑trained Q‑net...")
        qnet.load_state_dict(torch.load(config.pretrained_qnet_path, map_location=device))
    target_qnet = build_qnet(config).to(device)
    target_qnet.load_state_dict(qnet.state_dict())
    target_qnet.eval()

    # --- Mixer ---
    mixer = build_mixer(n_agents=config.max_gen_length,
                        feat_dim=config.state_dim,
                        config=config).to(device)
    if config.use_pretrained_mixer:
        print("Loading pre‑trained mixer...")
        mixer.load_state_dict(torch.load(config.pretrained_mixer_path, map_location=device))
    target_mixer = build_mixer(n_agents=config.max_gen_length,
                               feat_dim=config.state_dim,
                               config=config).to(device)
    target_mixer.load_state_dict(mixer.state_dict())
    target_mixer.eval()

    # --- Reward Model ---
    rm = RewardModel(config).to(device)
    if config.use_pretrained_rm:
        print("Loading pre‑trained RM...")
        rm.load_state_dict(torch.load(config.pretrained_rm_path, map_location=device))

    if not config.train_rm:
        rm.eval()
        for p in rm.parameters():
            p.requires_grad = False
        target_rm = None
    else:
        rm_optimizer = optim.AdamW(rm.parameters(), lr=config.rm_lr, weight_decay=config.weight_decay)
        rm_scheduler = optim.lr_scheduler.CosineAnnealingLR(rm_optimizer, T_max=config.episodes, eta_min=1e-6)
        rm_buffer = []
        rm_capacity = config.rm_buffer_capacity

        target_rm = RewardModel(config).to(device)
        target_rm.load_state_dict(rm.state_dict())
        target_rm.eval()
        for p in target_rm.parameters():
            p.requires_grad = False

    # --- Matcher ---
    matcher = BipartiteMatcher(config)

    # --- Policy Optimizer ---
    trainable_params = list(qnet.parameters()) + list(mixer.parameters())
    if config.train_rm:
        trainable_params += list(rm.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.episodes, eta_min=1e-6)

    # --- Replay Buffer ---
    buffer = PrioritizedReplayBuffer(
        capacity=config.buffer_capacity,
        alpha=config.per_alpha,
        beta=config.per_beta,
        beta_increment=config.per_beta_increment
    )

    epsilon = config.epsilon_start

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=config.log_dir)

    csv_path = os.path.join(config.log_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "reward", "steps", "epsilon", "lr", "loss", "episode_correct",
        "eval_acc", "eval_avg_steps", "eval_avg_reward", "rm_loss", "rm_acc",
        "q_grad_norm", "mixer_grad_norm", "curriculum_K", "warmup_steps"
    ])

    best_accuracy = 0.0
    best_model_path = os.path.join(config.save_dir, "best_model.pth")
    best_rm_path = os.path.join(config.save_dir, "best_rm.pth")
    best_target_rm_path = os.path.join(config.save_dir, "best_target_rm.pth")

    potential_rm_weight = config.potential_rm_weight
    potential_entropy_weight = config.potential_entropy_weight

    episodes_per_problem = config.episodes_per_problem
    problem_ep_counter = 0
    current_idx = np.random.randint(num_train)

    episode_pbar = tqdm(range(1, config.episodes + 1), desc="Training episodes", unit="ep")
    for episode in episode_pbar:
        # ---- Curriculum learning: compute K (number of heuristic warmup steps) ----
        if config.use_curriculum:
            if config.curriculum_decay_type == 'linear':
                decay_per_ep = config.curriculum_K_start / config.curriculum_decay_episodes
                K_max = max(config.curriculum_K_end,
                            int(config.curriculum_K_start - decay_per_ep * episode))
            elif config.curriculum_decay_type == 'exp':
                ratio = (config.curriculum_K_end / max(config.curriculum_K_start, 1)) ** (1.0 / config.curriculum_decay_episodes)
                K_max = max(config.curriculum_K_end,
                            int(config.curriculum_K_start * (ratio ** episode)))
            else:
                K_max = 0

            if config.curriculum_random_sample and K_max > 0:
                K_random = np.random.randint(0, K_max + 1)
            else:
                K_random = K_max

            if np.random.rand() < config.curriculum_force_random_prob:
                K_random = 0

            if config.curriculum_per_window_mode:
                K = max(1, round(K_random / num_windows)) if K_random > 0 else 0
            else:
                K = K_random
        else:
            K = 0

        q_grad_norm = 0.0
        mixer_grad_norm = 0.0

        # ---- Sample a problem (repeat episodes_per_problem times) ----
        if problem_ep_counter >= episodes_per_problem:
            current_idx = np.random.randint(num_train)
            problem_ep_counter = 0
        problem_ep_counter += 1

        prompt = problems[current_idx]
        ref_answer = references[current_idx]["answer"]
        ref_reasoning = references[current_idx]["reasoning"]

        state = env.reset(prompt, ref_answer, ref_reasoning)
        agent_mask = env.agent_mask.to(device)
        attention_mask = env.attention_mask.to(device)
        done = False
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        last_info = {}
        warmup_steps = 0

        if config.train_rm:
            rm_states = []

        # ---- Episode loop ----
        while not done:
            # Decide whether to use heuristic or RL for this step
            if config.use_curriculum:
                if config.curriculum_per_window_mode:
                    use_heuristic = (env.steps_in_current_window < K)
                else:
                    use_heuristic = (episode_steps < K)
            else:
                use_heuristic = False

            if use_heuristic:
                action = get_heuristic_action(env, config)
                warmup_steps += 1
            else:
                q_adv, v_global = qnet(state.to(device), agent_mask)
                is_mask = env.is_mask.to(device)
                can_unmask = env.can_unmask.to(device)
                can_mask = env.can_mask.to(device)
                action = matcher.select_action(q_adv, is_mask, agent_mask,
                                               can_unmask, can_mask, epsilon,
                                               step_idx=episode_steps)

            next_state, env_reward, done, info = env.step(action)
            last_info = info
            total_reward = env_reward

            if config.train_rm:
                rm_states.append((next_state.cpu(), attention_mask.cpu()))

            buffer.push(state, action, total_reward, next_state, done,
                        agent_mask.cpu(), attention_mask.cpu(), step=episode_steps)

            state = next_state
            episode_reward += total_reward
            episode_steps += 1

            # ---- Training step if buffer has enough samples ----
            if len(buffer) >= config.batch_size:
                batch = buffer.sample(config.batch_size)
                if batch is not None:
                    (states_b, actions_b, rewards_b, next_states_b, dones_b,
                     agent_masks_b, attn_masks_b, steps_b, indices, weights) = batch

                    states_b = states_b.to(device)
                    next_states_b = next_states_b.to(device)
                    actions_b = actions_b.to(device)
                    rewards_b = rewards_b.to(device)
                    dones_b = dones_b.to(device)
                    agent_masks_b = agent_masks_b.to(device)
                    attn_masks_b = attn_masks_b.to(device)
                    steps_b = steps_b.to(device)
                    weights = weights.to(device)

                    # Extract is_mask from state (index 3)
                    next_is_mask_b = next_states_b[..., 3].bool()

                    # Extract permission masks from state (indices as computed)
                    can_unmask_b = states_b[..., can_unmask_idx].bool()
                    can_mask_b = states_b[..., can_mask_idx].bool()
                    next_can_unmask_b = next_states_b[..., can_unmask_idx].bool()
                    next_can_mask_b = next_states_b[..., can_mask_idx].bool()

                    with torch.no_grad():
                        # PBRS: potential-based reward from RM
                        if potential_rm_weight != 0.0 and target_rm is not None:
                            p_cur = target_rm(states_b, attn_masks_b).float()
                            p_next = target_rm(next_states_b, attn_masks_b).float()
                            p_next = p_next * (1.0 - dones_b) + dones_b * (rewards_b > 0.0).float()
                            rm_reward_b = config.gamma * p_next - p_cur
                        else:
                            rm_reward_b = 0.0

                        # Entropy gain bonus (optional)
                        if potential_entropy_weight != 0.0:
                            entropy_cur = states_b[..., entropy_idx]
                            entropy_next = next_states_b[..., entropy_idx]
                            agent_float = agent_masks_b.float()
                            cur_mean_entropy = (entropy_cur * agent_float).sum(dim=1) / agent_float.sum(dim=1).clamp(min=1)
                            next_mean_entropy = (entropy_next * agent_float).sum(dim=1) / agent_float.sum(dim=1).clamp(min=1)
                            entropy_gain = cur_mean_entropy - config.gamma * next_mean_entropy
                            entropy_gain = entropy_gain.unsqueeze(1)
                        else:
                            entropy_gain = 0.0

                        total_reward_b = rewards_b \
                                         + potential_rm_weight * rm_reward_b \
                                         + potential_entropy_weight * entropy_gain

                    # ---- Current Q-values ----
                    q_adv_all, v_all = qnet(states_b, agent_masks_b)

                    # Taken advantages for the batch actions
                    adv_taken = q_adv_all.gather(2, actions_b.unsqueeze(-1)).squeeze(-1)
                    adv_taken = adv_taken * agent_masks_b.to(adv_taken.dtype)

                    # Dueling baseline: approximate best advantage (fast, not exact)
                    if config.use_dueling_mixer:
                        # Simple approximation: max over actions weighted by permission
                        best_cur_adv = q_adv_all[..., 1] * can_unmask_b.float() / 2 + \
                                       q_adv_all[..., 0] * (1 - can_unmask_b.float()) / 2 + \
                                       q_adv_all[..., 0] * can_mask_b.float() / 2 + \
                                       q_adv_all[..., 1] * (1 - can_mask_b.float()) / 2
                        best_cur_adv = best_cur_adv * agent_masks_b.to(best_cur_adv.dtype)
                        B_cur = best_cur_adv.sum(dim=1, keepdim=True) / agent_masks_b.sum(dim=1, keepdim=True)
                    else:
                        B_cur = 0.0

                    # Auxiliary loss for monotonicity (QMIX only)
                    if config.aux_lambda != 0:
                        with torch.no_grad():
                            q_adv_detached = q_adv_all.detach()
                            rand_actions = torch.randint(0, 2, actions_b.shape, device=device)
                            rand_actions = torch.where(can_unmask_b & can_mask_b, rand_actions,
                                                       torch.where(can_unmask_b & (~can_mask_b),
                                                                   torch.ones_like(rand_actions),
                                                                   torch.zeros_like(rand_actions)))
                            q_rand_adv = q_adv_detached.gather(2, rand_actions.unsqueeze(-1)).squeeze(-1)
                            q_rand_values = q_rand_adv * agent_masks_b.float()
                    else:
                        q_rand_values = None

                    mixer_result = mixer(adv_taken, states_b, attn_masks_b, agent_masks_b,
                                         q_values_2=q_rand_values)
                    if isinstance(mixer_result, tuple):
                        adv_mean, aux_loss = mixer_result
                    else:
                        adv_mean = mixer_result
                        aux_loss = 0.0

                    q_tot_raw = adv_mean + v_all
                    q_tot = q_tot_raw - B_cur

                    # ---- Target Q-tot with Double DQN and dueling baseline ----
                    with torch.no_grad():
                        # Next advantages from online Q-net (for action selection)
                        q_next_adv, _ = qnet(next_states_b, agent_masks_b)
                        # Select actions using online Q-net and matcher
                        next_actions_list = []
                        B_curr = config.batch_size
                        for i in range(B_curr):
                            next_step_idx = (steps_b[i] + 1).item()
                            act = matcher.select_action(
                                q_next_adv[i:i+1],
                                next_is_mask_b[i:i+1],
                                agent_masks_b[i:i+1],
                                next_can_unmask_b[i:i+1],
                                next_can_mask_b[i:i+1],
                                epsilon=0.0,
                                step_idx=next_step_idx
                            )
                            next_actions_list.append(act)
                        next_actions = torch.cat(next_actions_list, dim=0)

                        # Evaluate selected actions using target Q-net
                        q_next_target_adv, v_next_target = target_qnet(next_states_b, agent_masks_b)
                        adv_next_taken = q_next_target_adv.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)

                        if config.use_dueling_mixer:
                            # Approximate best advantage for next state
                            best_next_adv = q_next_target_adv[..., 1] * next_can_unmask_b.float() / 2 + \
                                            q_next_target_adv[..., 0] * (1 - next_can_unmask_b.float()) / 2 + \
                                            q_next_target_adv[..., 0] * next_can_mask_b.float() / 2 + \
                                            q_next_target_adv[..., 1] * (1 - next_can_mask_b.float()) / 2
                            best_next_adv = best_next_adv * agent_masks_b.to(best_next_adv.dtype)
                            B_next = best_next_adv.sum(dim=1, keepdim=True) / agent_masks_b.sum(dim=1, keepdim=True).clamp(min=1)
                        else:
                            B_next = 0.0

                        adv_next_taken = adv_next_taken * agent_masks_b.to(adv_next_taken.dtype)
                        target_adv_mean = target_mixer(adv_next_taken, next_states_b,
                                                       attn_masks_b, agent_masks_b)
                        target_q_tot_raw = target_adv_mean + v_next_target
                        target_q_tot = target_q_tot_raw - B_next

                        td_target = total_reward_b + config.gamma * target_q_tot * (1.0 - dones_b)

                    # ---- Loss and optimization ----
                    huber = F.smooth_l1_loss(q_tot, td_target, reduction='none', beta=config.huber_delta)
                    main_loss = (weights.unsqueeze(1) * huber).mean()
                    loss = main_loss + config.aux_lambda * aux_loss

                    optimizer.zero_grad()
                    loss.backward()

                    # Compute gradient norms for logging
                    q_grad_norm = 0.0
                    for p in qnet.parameters():
                        if p.grad is not None:
                            q_grad_norm += p.grad.data.norm(2).item() ** 2
                    q_grad_norm = q_grad_norm ** 0.5

                    mixer_grad_norm = 0.0
                    for p in mixer.parameters():
                        if p.grad is not None:
                            mixer_grad_norm += p.grad.data.norm(2).item() ** 2
                    mixer_grad_norm = mixer_grad_norm ** 0.5

                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=config.grad_clip_norm
                    )
                    optimizer.step()

                    # Update priorities in PER
                    with torch.no_grad():
                        abs_td = (q_tot - td_target).abs().detach().cpu().numpy()
                    buffer.update_priorities(indices, abs_td.flatten())

                    # Soft update target networks
                    for param, target_param in zip(qnet.parameters(), target_qnet.parameters()):
                        target_param.data.copy_(config.tau * param.data + (1.0 - config.tau) * target_param.data)
                    for param, target_param in zip(mixer.parameters(), target_mixer.parameters()):
                        target_param.data.copy_(config.tau * param.data + (1.0 - config.tau) * target_param.data)

                    episode_losses.append(loss.item())

        # ---- Episode finished ----
        episode_correct = float(last_info.get('current_correct', 0.0))
        target_value = 1.0 if episode_correct else 0.0

        # ---- Update Reward Model (Focal Loss) ----
        rm_loss_val = 0.0
        rm_acc_val = 0.0
        if config.train_rm:
            for state_cpu, attn_cpu in rm_states:
                if len(rm_buffer) >= rm_capacity:
                    rm_buffer.pop(0)
                rm_buffer.append((state_cpu, attn_cpu, target_value))

            if len(rm_buffer) >= config.rm_batch_size and episode % config.rm_update_interval == 0:
                all_targets = [item[2] for item in rm_buffer]
                pos_ratio = np.mean(all_targets)
                ema_pos_ratio = 0.9 * ema_pos_ratio + 0.1 * pos_ratio

                all_states = [item[0] for item in rm_buffer]
                all_attns = [item[1] for item in rm_buffer]
                all_targets = [item[2] for item in rm_buffer]
                state_buffer = torch.cat(all_states, dim=0)
                attn_buffer = torch.cat(all_attns, dim=0)
                target_buffer = torch.tensor(all_targets, dtype=torch.float32, device=device).unsqueeze(1)

                total_rm_loss = 0.0
                total_rm_acc = 0.0
                num_updates = 0

                rm.train()
                for _ in range(config.rm_epochs):
                    indices = torch.randperm(len(rm_buffer))
                    for start in range(0, len(rm_buffer), config.rm_batch_size):
                        end = min(start + config.rm_batch_size, len(rm_buffer))
                        batch_idx = indices[start:end]

                        state_batch = state_buffer[batch_idx].to(device)
                        attn_batch = attn_buffer[batch_idx].to(device)
                        target_batch = target_buffer[batch_idx]

                        pred = rm(state_batch, attn_batch).float()
                        if config.rm_focal_alpha is None:
                            rm_focal_alpha = max(0.1, 1 - ema_pos_ratio)
                        else:
                            rm_focal_alpha = config.rm_focal_alpha
                        loss_rm = focal_loss(pred, target_batch,
                                             alpha=rm_focal_alpha,
                                             gamma=config.rm_focal_gamma)
                        rm_optimizer.zero_grad()
                        loss_rm.backward()
                        torch.nn.utils.clip_grad_norm_(
                            rm.parameters(),
                            max_norm=config.grad_clip_norm
                        )
                        rm_optimizer.step()

                        total_rm_loss += loss_rm.item()
                        pred_class = (pred > 0.5).float()
                        total_rm_acc += (pred_class == target_batch).float().mean().item()
                        num_updates += 1

                rm.eval()
                rm_loss_val = total_rm_loss / max(num_updates, 1)
                rm_acc_val = total_rm_acc / max(num_updates, 1)

                with torch.no_grad():
                    for target_param, param in zip(target_rm.parameters(), rm.parameters()):
                        target_param.data.copy_(config.rm_tau * param.data + (1.0 - config.rm_tau) * target_param.data)

        # ---- End of episode updates ----
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        writer.add_scalar("Train/Reward", episode_reward, episode)
        writer.add_scalar("Train/Steps", episode_steps, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        writer.add_scalar("Train/LR", current_lr, episode)
        writer.add_scalar("Train/Loss", avg_loss, episode)
        writer.add_scalar("Train/Correct", episode_correct, episode)
        writer.add_scalar("Train/RM_Loss", rm_loss_val, episode)
        writer.add_scalar("Train/RM_Accuracy", rm_acc_val, episode)
        writer.add_scalar("Train/Q_GradNorm", q_grad_norm, episode)
        writer.add_scalar("Train/Mixer_GradNorm", mixer_grad_norm, episode)
        writer.add_scalar("Train/CurriculumK", K, episode)
        writer.add_scalar("Train/WarmupSteps", warmup_steps, episode)

        csv_writer.writerow([
            episode, episode_reward, episode_steps, epsilon, current_lr, avg_loss, episode_correct,
            "", "", "", rm_loss_val, rm_acc_val, q_grad_norm, mixer_grad_norm, K, warmup_steps
        ])

        episode_pbar.set_postfix({
            "reward": f"{episode_reward:.3f}",
            "steps": episode_steps,
            "eps": f"{epsilon:.3f}",
            "loss": f"{avg_loss:.4f}",
            "corr": f"{episode_correct:.0f}",
            "rm_l": f"{rm_loss_val:.3f}",
            "buf": len(buffer),
            "K": K
        })

        # ---- Evaluation ----
        if episode % config.eval_interval == 0:
            temp_path = os.path.join(config.save_dir, f"qnet_ep{episode}.pth")
            torch.save(qnet.state_dict(), temp_path)
            eval_results = evaluate(
                checkpoint_path=temp_path, model=model, tokenizer=tokenizer,
                max_samples=config.fast_eval_samples
            )
            acc = eval_results["accuracy"]
            avg_steps = eval_results["avg_steps"]
            avg_reward = eval_results["avg_reward"]

            tqdm.write(f"Ep {episode}: Acc={acc:.4f}, AvgSteps={avg_steps:.2f}, AvgReward={avg_reward:.4f}")
            writer.add_scalar("Eval/Accuracy", acc, episode)
            writer.add_scalar("Eval/AvgSteps", avg_steps, episode)
            writer.add_scalar("Eval/AvgReward", avg_reward, episode)

            csv_writer.writerow([episode, "", "", "", "", "", "", acc, avg_steps, avg_reward, "", "", "", "", "", ""])

            if acc > best_accuracy:
                best_accuracy = acc
                torch.save(qnet.state_dict(), best_model_path)
                if config.train_rm:
                    torch.save(rm.state_dict(), best_rm_path)
                    torch.save(target_rm.state_dict(), best_target_rm_path)
                tqdm.write(f"  New best model saved with accuracy {acc:.4f}")

        # ---- Save checkpoint ----
        if episode % config.save_interval == 0:
            save_path = os.path.join(config.save_dir, f"qnet_ep{episode}.pth")
            torch.save(qnet.state_dict(), save_path)
            if config.train_rm:
                torch.save(rm.state_dict(), os.path.join(config.save_dir, f"rm_ep{episode}.pth"))
            tqdm.write(f"Checkpoint saved to {save_path}")

    csv_file.close()
    writer.close()

    print("Training finished.")
    print("Running final evaluation...")
    final_path = os.path.join(config.save_dir, "qnet_final.pth")
    torch.save(qnet.state_dict(), final_path)
    if config.train_rm:
        torch.save(rm.state_dict(), os.path.join(config.save_dir, "rm_final.pth"))
        torch.save(target_rm.state_dict(), os.path.join(config.save_dir, "target_rm_final.pth"))
    final_results = evaluate(checkpoint_path=final_path, model=model, tokenizer=tokenizer)
    print(f"Final Accuracy: {final_results['accuracy']:.4f}, "
          f"Avg Steps: {final_results['avg_steps']:.2f}, "
          f"Avg Reward: {final_results['avg_reward']:.4f}")


if __name__ == "__main__":
    train()