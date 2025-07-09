from argument import parse_arg, build_hf_config
from alignment.model_utils import get_ppo_policy_model_pair, get_hf_tokenizer, \
                                        load_policy_into_vllm_instance, update_old_policy, \
                                        update_vllm_model_for_rft, get_vllm_sampling_param, get_vllm_models_with_sampling_param
from alignment.dataset import get_train_data_loader, get_test_data_loader
from alignment.optimizer import get_optimizer
from alignment.rl_utils import compute_group_normalized_rewards, grpo_training_loop
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.grpo_ray_utils import GRPO_Trainer, VLLM_Generator
from math_verify import parse
from tqdm import tqdm
from evaluate import evaluate_vllm
from collections import defaultdict
from utils import row_to_col_oriented, save_hf_model
import torch
import ray
import time


def rft(args):
    train_loader = get_train_data_loader(args, tokenizer=None)

    policy_model, old_policy_model = get_ppo_policy_model_pair(args)
    tokenizer = get_hf_tokenizer(args)
    optimizer = get_optimizer(args, policy_model)
    
    vllm, sampling_params = get_vllm_models_with_sampling_param(args)

    policy_model.to(torch.device("cuda:1"))
    old_policy_model.to(torch.device("cuda:1"))
    policy_model.train()
    lora_request = None
 
    progress_bar = tqdm(
        train_loader,
        total=args.training_step,
        desc="RFT Training")
    for idx, data in enumerate(progress_bar):
        questions = data["questions"]
        gt = [parse(a)[-1] for a in data["answers"]]
        # repeat the gt answers to match the number of samples per prompt
        repeated_gt = [a for a in gt for _ in range(args.num_samples_per_prompt)]

        # vllm_flag = vllm.generate.remote(questions, sampling_params, use_tqdm=False, lora_request=lora_request)
        vllm_outputs = vllm.generate(questions, sampling_params, use_tqdm=False, lora_request=lora_request)
        policy_rollouts = []
        for output in vllm_outputs:
            policy_rollouts.extend([p.text for p in output.outputs])

        train_start = time.perf_counter()
        advantages, raw_rewards, rw_infos = compute_group_normalized_rewards(args, r1_zero_reward_fn, policy_rollouts, repeated_gt)
        training_infos = grpo_training_loop(args, questions, policy_rollouts, advantages, policy_model, old_policy_model, tokenizer, optimizer)
        train_end = time.perf_counter()

        update_old_policy(args, policy_model, old_policy_model)
        lora_request = update_vllm_model_for_rft(args, policy_model, vllm, lora_adapter_name="new_policy", lora_int_id=idx+1)
        end = time.perf_counter()

        log_infos = {
            "iter_time": f"{(train_end - train_start):.2f}",
            "ckpt_update_time": f"{(end - train_end):.2f}"
        }
        log_infos.update(rw_infos)
        log_infos.update(training_infos)
        progress_bar.set_postfix(log_infos)
        

        if (idx + 1) >= args.training_step:
            break

    print("evaluating...")
    test_loader = get_test_data_loader(args, tokenizer=None)
    test_data = row_to_col_oriented(test_loader)
    eval_sampling_params = get_vllm_sampling_param(args, n=1)
    evaluate_vllm(vllm, r1_zero_reward_fn, test_data, eval_sampling_params, lora_request=lora_request)

    print("saving trained model...")
    save_hf_model(args, policy_model, tokenizer, exp_name="RFT", adapter_names=["new_policy"])
    print("experiment done")
    
if __name__ == "__main__":
    
    args = parse_arg()
    build_hf_config(args)
    rft(args)
