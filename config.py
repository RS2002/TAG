# config.py
# Central configuration for the TAG project.
# All hyper‑parameters are explicitly defined here.

import torch

class Config:
    # ==================== Paths ====================
    model_path = "../LLaDA-8B-Instruct"
    data_path = "../grade-school-math/grade_school_math/data"
    train_file = "train.jsonl"
    test_file = "test.jsonl"
    save_dir = "./checkpoints"
    log_dir = "./logs"

    # ==================== Model Loading ====================
    model_dtype = torch.bfloat16
    device_map = "cuda:0"
    freeze_backbone = True

    # ==================== Sequence & Inference ====================
    max_total_length = 364            # total sequence length after left padding (including prompt + generation)
    max_gen_length = 128              # length of the generation region (agent tokens)
    block_length = 364                # block length for block‑wise generation (unused)
    max_mask_steps = 128              # maximum denoising steps per episode
    generation_temperature = 0.0      # 0 = argmax, >0 enables Gumbel sampling

    # ==================== State Construction ====================
    use_hidden_state = False          # include LLaDA hidden states in the state tensor
    top_k_conf = 5                    # number of top‑k probabilities to include
    d_model = 4096                    # LLaDA hidden dimension
    hidden_proj_dim = 256             # projection dim for hidden states (if used)
    global_feat_dim = 14              # dimension of global features (mask_count, unmask_count,
                                      #   mean_entropy, mean_action, step_norm, can_unmask, can_mask,
                                      #   conf_rank, ent_rank, is_eos_pred, is_eos_current,
                                      #   agent_idx, attn_score, attn_rank)

    # ==================== Q‑Network Architecture ====================
    transformer_heads = 4
    transformer_layers = 2
    other_proj_dim = 256              # projection dimension for non‑hidden features

    # ==================== Reward Model (RM) Architecture ====================
    rm_heads = 4
    rm_layers = 2
    rm_lr = 3e-4
    rm_update_interval = 1            # update RM every N episodes
    rm_epochs = 1
    rm_tau = 1.0                      # target RM soft‑update factor
    rm_batch_size = 128
    rm_buffer_capacity = 3000

    # ==================== Mixer Architecture ====================
    mixer_type = 'vdn'                # 'vdn' or 'qmix'
    mixer_embed_dim = 64
    mixer_transformer_dim = 64        # used only when mixer_type='qmix'
    mixer_transformer_heads = 4
    mixer_transformer_layers = 2

    # ==================== Matching Constraint ====================
    k_min = 1                         # minimum masks to remove per step (original mode)

    # ==================== Denoising Mode ====================
    denoising_mode = 'original'       # 'original' or 'linear_cap'
    linear_cap_k = 1                  # mask upper bound reduction per step (linear_cap)

    # ==================== Reward Coefficients ====================
    alpha_speed = 0.0                 # speed reward exponent (0 = disabled)

    # ==================== Potential Function Coefficients ====================
    potential_rm_weight = 0.0        # weight for RM‑based PBRS (0 = disabled)
    potential_entropy_weight = 0.0    # weight for entropy gain bonus (0 = disabled)

    # ==================== Curriculum Learning ====================
    use_curriculum = False
    curriculum_K_start = 32
    curriculum_K_end = 0
    curriculum_decay_type = 'linear'
    curriculum_decay_episodes = 5000
    curriculum_random_sample = True   # sample K ~ Uniform(0, K_max)
    curriculum_min_unmask = 1
    curriculum_max_unmask = 1
    curriculum_force_random_prob = 0.1
    curriculum_per_window_mode = False # distribute warmup steps per sliding window

    # ==================== Pre‑trained Model Loading ====================
    use_pretrained_qnet = False
    pretrained_qnet_path = ""
    use_pretrained_mixer = False
    pretrained_mixer_path = ""
    use_pretrained_rm = False
    pretrained_rm_path = ""
    train_rm = False

    # ==================== Training Hyperparameters ====================
    lr = 3e-4
    gamma = 0.99
    tau = 0.005                       # target network EMA factor
    buffer_capacity = 10000
    batch_size = 128
    episodes = 5000
    eval_interval = 100
    save_interval = 500
    episodes_per_problem = 1
    fast_eval_samples = 20

    # ==================== Exploration ====================
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.999

    # ==================== Prioritized Experience Replay ====================
    use_per = True
    per_alpha = 0.6
    per_beta = 0.4
    per_beta_increment = 0.001

    # ==================== Auxiliary Loss (QMIX) ====================
    aux_lambda = 0.0

    # ==================== Gradient Clipping ====================
    grad_clip_norm = 1.0

    # ==================== Miscellaneous ====================
    seed = 42
    num_workers = 4

    # ==================== Sliding Window ====================
    use_sliding_window = False
    window_size = 128
    window_stride = 128

    # ==================== Dueling Architecture ====================
    use_dueling_mixer = False
    dueling_hidden_dim = 256

    # ==================== Loss & Regularization ====================
    huber_delta = 1.0
    weight_decay = 1e-4

    # ==================== RM Loss (Focal) ====================
    rm_focal_alpha = 0.5
    rm_focal_gamma = 2.0

    # ==================== History & Attention Features ====================
    history_len = 1                   # number of past states concatenated (1 = no history)
    use_attn_features = True          # enable attention‑based features
    attn_target_layers = [15]         # which LLaDA layers to average for attention
    attn_unmask_ratio = 0.2           # fraction of top attention‑ranked eligible tokens to keep unmaskable
    attn_unmask_min_keep = 1          # minimum number of tokens to keep (prevents zero)