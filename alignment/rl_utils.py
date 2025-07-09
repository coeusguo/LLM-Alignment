import math
import torch
import argparse
from collections import defaultdict
from typing import Callable, Tuple, Literal, Union
from transformers import PreTrainedModel, AutoTokenizer
from .sft_utils import tokenize_prompt_and_output, get_response_log_probs, reduce_masked_entropy
from .profiling import print_gpu_memory

def compute_group_normalized_rewards(
    args: argparse.Namespace,
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_gt: list[str]
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, str]]:

    group_size = args.num_samples_per_prompt
    advantage_eps = args.advantage_eps
    normalize_by_std = args.normalize_by_std
    
    assert len(rollout_responses) == len(repeated_gt)
    assert len(rollout_responses) % group_size == 0

    rewards = []
    for idx in range(len(rollout_responses)):
        reward = reward_fn(rollout_responses[idx], repeated_gt[idx])["reward"]
        rewards.append(reward)

    rewards = torch.tensor(rewards, dtype=torch.float32)
    grouped_rewards = rewards.reshape(-1, group_size)
    group_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    group_normalized_rewards = grouped_rewards - group_mean

    if normalize_by_std:
        assert advantage_eps is not None
        group_std = grouped_rewards.std(dim=-1, keepdim=True)
        group_normalized_rewards = group_normalized_rewards / (group_std + advantage_eps)

    infos = {
        "mean_reward": f"{torch.mean(rewards.detach()).item():.2f}",
    }
    return group_normalized_rewards.reshape(-1), rewards, infos

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # (batch_size, 1)
    policy_log_probs: torch.Tensor # (batch_size, seq_len)
) -> torch.Tensor:
    raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1)
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)

    advantages = advantages.view(-1, 1)
    positive_adv_mask = advantages > 0
    upper_bound = torch.full_like(ratio, 1 + clip_range)
    lower_bound = torch.full_like(ratio, 1 - clip_range)
    cliped_ratio = torch.where(positive_adv_mask, 
                            torch.min(ratio, upper_bound), torch.max(ratio, lower_bound))

    loss = compute_naive_policy_gradient_loss(advantages, cliped_ratio)
    pos_cliped_mask = ((ratio > 1 + clip_range) & positive_adv_mask) 
    neg_clipped_mask = ((ratio < 1 - clip_range) & ~positive_adv_mask)

    pos_rate = positive_adv_mask.sum() / positive_adv_mask.numel()
    neg_rate = (~positive_adv_mask).sum() / positive_adv_mask.numel()
    token_pos_cls = ratio.numel() * pos_rate
    token_neg_cls = ratio.numel() * neg_rate

    infos = {
        "pos_clip_frac": pos_cliped_mask.sum() / (token_pos_cls+ 1e-5),
        "neg_clip_frac": neg_clipped_mask.sum() / (token_neg_cls + 1e-5),
    }
    return loss, infos

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    clip_range: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    infos = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        loss, infos = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    return loss, infos

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None
) -> torch.Tensor:
    tensor = tensor * mask
    denom = torch.sum(mask, dim=dim)
    return tensor.sum(dim=dim) / denom

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_step: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> Tuple[torch.Tensor, dict[str, float]]:
    per_token_loss, infos = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    
    loss = masked_mean(per_token_loss, response_mask, dim=-1).mean()
    loss = loss / gradient_accumulation_step
    all_infos = {}
    all_infos.update(infos)
    loss.backward()

    return loss, all_infos

def grpo_training_loop(
        args: argparse.Namespace,
        question_prompts: list[str],
        rollout_responses: list[str],
        advantages: torch.Tensor,
        policy_llm: PreTrainedModel,
        old_policy_llm: PreTrainedModel,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        prompt_format_func = None
) -> dict[str, str]:
    
    """
    perform 1 batch grpo updates
    """
    group_size = args.grpo_group_size
    repeated_question_prompts = [p for p in question_prompts for _ in range(group_size)] 
    assert len(repeated_question_prompts) == len(rollout_responses)
    total_samples = len(rollout_responses)

    device = next(policy_llm.parameters()).device
    tokens = tokenize_prompt_and_output(repeated_question_prompts, 
                               rollout_responses, tokenizer, prompt_format_func)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    advantages = advantages.to(device)

    # generate old policy log probs all at once
    with torch.no_grad():
        old_policy_log_probs = get_response_log_probs(old_policy_llm, tokens["input_ids"], 
                                                      tokens["labels"], return_token_entropy=False)["log_probs"]

    micro_rollout_batch_size = args.rollout_batch_size // args.accumulation_steps
    num_rollout_micro_batches = total_samples // micro_rollout_batch_size

    log_infos = defaultdict(float)
    for epoch in range(args.epochs_per_rollout_batch):
        for idx in range(num_rollout_micro_batches):
            batch_start = idx * micro_rollout_batch_size
            batch_end = (idx + 1) * micro_rollout_batch_size
            micro_batch_input = tokens["input_ids"][batch_start:batch_end, :]
            micro_batch_labels = tokens["labels"][batch_start:batch_end, :]
            micro_batch_mask = tokens["response_mask"][batch_start:batch_end, :]
            micro_advantages = advantages[batch_start:batch_end]

            micro_old_policy_log_probs = old_policy_log_probs[batch_start:batch_end, :]
            policy_out = get_response_log_probs(policy_llm, micro_batch_input, micro_batch_labels, return_token_entropy=True, chunkify=None)
            loss, loss_infos = grpo_microbatch_train_step(policy_out["log_probs"], micro_batch_mask, args.accumulation_steps, 
                                       loss_type="grpo_clip", advantages=micro_advantages, old_log_probs=micro_old_policy_log_probs,
                                       cliprange=args.ppo_clip_range)

            log_infos["entropy"] += reduce_masked_entropy(policy_out["token_entropy"], micro_batch_mask).detach().item()
            log_infos.update({k: log_infos[k] + loss_infos[k] for k in loss_infos.keys()})
            if (idx + 1) % args.accumulation_steps  == 0:
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(policy_llm.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

    return {
        k: f"{v / num_rollout_micro_batches:.2f}" for k, v in log_infos.items()
    }

