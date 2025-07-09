from argument import parse_arg, build_hf_config
from alignment.model_utils import get_ppo_policy_model_pair, get_hf_tokenizer, \
                                        load_policy_into_vllm_instance, update_old_policy, \
                                        update_vllm_model_for_rft, get_vllm_sampling_param
from alignment.dataset import get_train_data_loader, get_test_data_loader
from alignment.optimizer import get_optimizer
from alignment.rl_utils import compute_group_normalized_rewards, grpo_training_loop
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.grpo_ray_utils import GRPO_Trainer, VLLM_Generator
from math_verify import parse
from tqdm import tqdm
from evaluate import evaluate_vllm
from collections import defaultdict, deque
from utils import row_to_col_oriented, save_hf_model
import torch
import ray
import time

# good enough to main async. ray training
BUFFER_SIZE = 2

def pre_process_data(data, num_samples_per_prompt):
    questions = data["questions"]
    gt = [parse(a)[-1] for a in data["answers"]]
    # repeat the gt answers to match the number of samples per prompt
    repeated_gt = [a for a in gt for _ in range(num_samples_per_prompt)]
    return questions, repeated_gt

def rft(args):
    train_loader = get_train_data_loader(args, tokenizer=None)

    # get vllm and grpo ray actors
    vllm = VLLM_Generator.remote(args)
    grpo_trainer = GRPO_Trainer.remote(args)
    
    # deploy ray actors
    ray.get([vllm.ready.remote(), grpo_trainer.ready.remote()])
    
    # create buffer queue to store actor flags
    vllm_buffer = deque(maxlen=BUFFER_SIZE)
    grpo_trainer_buffer = deque(maxlen=BUFFER_SIZE)

    progress_bar = tqdm(
        total=args.training_step,
        desc="RFT Training")

    training_iter = 0
    lora_request = None
    train_loader = iter(train_loader)

    # vllm buffer prefill step
    while len(vllm_buffer) < BUFFER_SIZE:
        data = next(train_loader)
        questions, repeated_gt = pre_process_data(data, args.num_samples_per_prompt)
        rollout_flag = vllm.rollout.remote(questions, use_tqdm=False)
        vllm_buffer.append((questions, repeated_gt, rollout_flag))

    while True:
        # terminal state check
        if training_iter + len(grpo_trainer_buffer) >= args.training_step:
            flags = [data[-1] for data in list(grpo_trainer_buffer)]

            # process the rest of the training steps
            while grpo_trainer_buffer:
                training_loop_flag, _, rw_infos = grpo_trainer_buffer.popleft()
                training_infos = ray.get(training_loop_flag)
                
                # update training log and step tqdm bar
                training_infos.update(rw_infos)
                progress_bar.set_postfix(training_infos)
                progress_bar.update(1)
            # break the endless while loop
            break

        # vllm actor
        if len(vllm_buffer) < BUFFER_SIZE:
            # make sure trainer at least trained once
            assert grpo_trainer_buffer, "something wrong happened"
            train_loop_flag, update_old_policy_flag, rw_infos = grpo_trainer_buffer.popleft()
            training_infos, policy_state_dict = ray.get([train_loop_flag, update_old_policy_flag])
            
            # update vllm policy model
            vllm.update_policy.remote(policy_state_dict)

            # preprocess data and genererate new prompts
            data = next(train_loader)
            questions, repeated_gt = pre_process_data(data, args.num_samples_per_prompt)
            rollout_flag = vllm.rollout.remote(questions, use_tqdm=False)

            # append to vllm buffer queue
            vllm_buffer.append((questions, repeated_gt, rollout_flag))

            # update training log and step tqdm bar
            training_infos.update(rw_infos)
            progress_bar.set_postfix(training_infos)
            progress_bar.update(1)
            training_iter += 1

        # grpo trainer actor
        if len(grpo_trainer_buffer) < BUFFER_SIZE:
            assert vllm_buffer, "something wrong happened"

            # pop out the oldest flag from vllm actor
            questions, repeated_gt, rollout_flag = vllm_buffer.popleft()
            policy_rollouts = ray.get(rollout_flag)

            # compute group normalized rewards
            advantages, raw_rewards, rw_infos = compute_group_normalized_rewards(args, r1_zero_reward_fn, policy_rollouts, repeated_gt)

            # execute the training loop and update old policy weights
            train_loop_flag = grpo_trainer.grpo_training_loop.remote(questions, policy_rollouts, advantages)
            update_old_policy_flag = grpo_trainer.update_old_policy.remote()

            grpo_trainer_buffer.append((train_loop_flag, update_old_policy_flag, rw_infos))


    print("evaluating...")
    test_loader = get_test_data_loader(args, tokenizer=None)
    test_data = row_to_col_oriented(test_loader)
    eval_sampling_params = get_vllm_sampling_param(args, n=1)
    ray.get(vllm.evaluate.remote(test_data, r1_zero_reward_fn, sampling_params=eval_sampling_params))
    
    print("saving trained model...")
    ray.get(grpo_trainer.save_policy_model.remote())
    print("experiment done")
    
if __name__ == "__main__":
    
    args = parse_arg()
    ray.init(num_gpus=2)
    build_hf_config(args)
    rft(args)
