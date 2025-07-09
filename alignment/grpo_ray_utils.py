import os
import ray
import torch
import shutil
import argparse
from collections import defaultdict
from transformers import PreTrainedModel, AutoTokenizer
from .sft_utils import get_response_log_probs, reduce_masked_entropy, tokenize_prompt_and_output
from .model_utils import get_vllm, get_ppo_policy_model_pair, get_vllm_sampling_param, update_old_policy, get_hf_tokenizer
from argument import build_hf_config, get_hf_config
from collections import OrderedDict
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from pathlib import Path, PosixPath
from .optimizer import get_optimizer
from .rl_utils import grpo_microbatch_train_step
from evaluate import evaluate_vllm
from utils import save_hf_model
from typing import Callable

# some hyper-params
POLICY_MODEL_NAME = "new_policy"
LORA_ADAPTER_NAME = "new_policy"
LORA_CACHE_FOLDER = "cache"


@ray.remote(num_gpus=1)
class GRPO_Trainer:
    def __init__(self,
        args: argparse.Namespace,
        prompt_format_fn = None
    ):
        build_hf_config(args)
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        self.args = args
        self.grpo_group_size = args.num_samples_per_prompt
        self.micro_rollout_batch_size = args.rollout_batch_size // args.accumulation_steps
        self.epochs_per_rollout_batch = args.epochs_per_rollout_batch
        self.accumulation_steps = args.accumulation_steps
        
        self.ppo_clip_range = args.ppo_clip_range
        self.max_grad_norm = args.max_grad_norm
        self.prompt_format_fn = prompt_format_fn

        # move policy model and old policy model to gpu
        self.policy_llm, self.old_policy_llm = get_ppo_policy_model_pair(args)
        # self.device = torch.device(f"cuda:{ray.get_gpu_ids()[0]}")
        self.device = torch.device("cuda:0")
        self.policy_llm.to(self.device)
        self.old_policy_llm.to(self.device)
        
        self.optimizer = get_optimizer(args, self.policy_llm)
        self.tokenizer = get_hf_tokenizer(args)

        # switch to train mode
        self.policy_llm.train() 

        self.lora_int_id = 1

        print(f"grpo trainer intiallized, current device: {ray.get_gpu_ids()}")

    def ready(self):
        return True

    def grpo_training_loop(self,
            question_prompts: list[str],
            rollout_responses: list[str],
            advantages: torch.Tensor,
    ) -> dict[str, str]:
        
        """
        perform 1 batch grpo updates
        """
        total_samples = len(rollout_responses)
        
        # e.g. [a, b, c] -> [a, a, b, b, c, c] for group_size = 2
        repeated_question_prompts = [p for p in question_prompts for _ in range(self.grpo_group_size)]
        assert len(repeated_question_prompts) == total_samples
        
        tokens = tokenize_prompt_and_output(repeated_question_prompts, 
                                rollout_responses, self.tokenizer, self.prompt_format_fn)
        
        # move data to GPU
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        advantages = advantages.to(self.device)

        # generate old policy log probs all at once
        with torch.no_grad():
            old_policy_log_probs = get_response_log_probs(self.old_policy_llm, tokens["input_ids"], 
                                                        tokens["labels"], return_token_entropy=False)["log_probs"]

        log_infos = defaultdict(float)
        for epoch in range(self.epochs_per_rollout_batch):
            for idx in range(self.accumulation_steps):

                # sclice into micro batches to decrease GPU memory consumption
                batch_start = idx * self.micro_rollout_batch_size
                batch_end = (idx + 1) * self.micro_rollout_batch_size
                micro_batch_input = tokens["input_ids"][batch_start:batch_end, :]
                micro_batch_labels = tokens["labels"][batch_start:batch_end, :]
                micro_batch_mask = tokens["response_mask"][batch_start:batch_end, :]
                micro_advantages = advantages[batch_start:batch_end]
                micro_old_policy_log_probs = old_policy_log_probs[batch_start:batch_end, :]

                policy_out = get_response_log_probs(self.policy_llm, micro_batch_input, micro_batch_labels, return_token_entropy=True, chunkify=None)
                loss, loss_infos = grpo_microbatch_train_step(policy_out["log_probs"], micro_batch_mask, self.accumulation_steps, 
                                        loss_type="grpo_clip", advantages=micro_advantages, old_log_probs=micro_old_policy_log_probs,
                                        cliprange=self.ppo_clip_range)

                loss_infos["entropy"] = \
                    reduce_masked_entropy(policy_out["token_entropy"], micro_batch_mask).detach().item() / self.accumulation_steps
                
                # accumulate log infos
                for k, v in loss_infos.items():
                    log_infos[k] = log_infos[k] + v

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_llm.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            k: f"{v / self.epochs_per_rollout_batch:.3f}" for k, v in log_infos.items()
        }
    def update_old_policy(self):
        
        update_old_policy(self.args, self.policy_llm, self.old_policy_llm)
        self.policy_llm.peft_model
        if self.args.lora:
            
            lora_path = Path(self.args.log_save_path) / f"{LORA_CACHE_FOLDER}_{self.lora_int_id}"
            os.makedirs(lora_path, exist_ok=True)
            self.policy_llm.peft_model.save_pretrained(lora_path, selected_adapters=[LORA_ADAPTER_NAME])

            self.lora_int_id += 1
            return lora_path

        return self.old_policy_llm.state_dict()

    def save_policy_model(self):
        save_hf_model(self.args, self.policy_llm, self.tokenizer, exp_name="RFT", adapter_names=[LORA_ADAPTER_NAME])


@ray.remote(num_gpus=1)
class VLLM_Generator:
    def __init__(self, args: argparse.Namespace):
        
        build_hf_config(args)
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.vllm = get_vllm(args)
        self.sampling_params = get_vllm_sampling_param(args)

        self.args = args
        self.lora_request = None
        self.lora_int_id = 1 # start from 1

        print(f"vllm intiallized, current device: {ray.get_gpu_ids()}")

    def rollout(self, prompts, sampling_params: None | SamplingParams = None, **kwargs) -> list[str]:
        if sampling_params is None:
            sampling_params = self.sampling_params

        vllm_outputs = self.vllm.generate(prompts, sampling_params, lora_request=self.lora_request, **kwargs)
        policy_rollouts = []
        for output in vllm_outputs:
            policy_rollouts.extend([p.text for p in output.outputs])
        return policy_rollouts

    def evaluate(self, 
                 prompts: dict[str, list[str]], 
                 reward_fn: Callable[[str, str], dict[str, float]], 
                 sampling_params: None | SamplingParams = None, 
                 lora_request: None | LoRARequest = None):

        '''
        need_to_parse_ground_truth: generally, gr
        '''
        if sampling_params is None:
            sampling_params = self.sampling_params
        if lora_request is None:
            lora_request = self.lora_request
        evaluate_vllm(self.vllm, reward_fn, prompts, sampling_params, lora_request=lora_request)

    def ready(self):
        return True

    def update_policy(self, state_dict: OrderedDict | PosixPath):
        """

        if state_dict is:
            OrderDict: the entire model will be updated
            str: policy model turned on lora, state_dict is the path to lora weights
        """
        if isinstance(state_dict, PosixPath):
            assert os.path.isdir(state_dict), f"you passed in a invalid path: {state_dict}"
            
            lora_path = state_dict / f"{LORA_ADAPTER_NAME}"
            # vllm will raise a warning if no tokenizer is found when loading the lora weight
            if not os.path.exists(lora_path / "tokenizer.json"):
                os.symlink(Path(get_hf_config().model_dir) / "tokenizer.json", lora_path / "tokenizer.json") 
            
            self.lora_request = LoRARequest(
                lora_name=LORA_ADAPTER_NAME,
                lora_local_path=str(lora_path),
                lora_int_id=self.lora_int_id)

            # remote previous cache folder to save disk memory (e.g. current: cache_4, remove: cache_1)
            useless_lora_path = Path(str(state_dict).replace(str(self.lora_int_id), str(self.lora_int_id - 3)))
            if os.path.isdir(useless_lora_path):
                shutil.rmtree(useless_lora_path)
            self.lora_int_id += 1

        elif isinstance(state_dict, OrderedDict):
            raise NotImplementedError(f"not yet implemented {type(state_dict)}")
        else:
            raise NotImplementedError(f"unsupported type {type(state_dict)}")