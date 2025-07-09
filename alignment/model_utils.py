from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from argument import get_hf_config
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from peft import get_peft_model, LoraConfig, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Tuple
from argument import get_hf_config
from collections import OrderedDict
import torch
import warnings
import time
import os

def unwrap_model(model):
    while hasattr(model, "model"):
        model = model.model
    return model

def get_vllm_sampling_param(args, n = None):
    return SamplingParams(
        n=n if n is not None else args.num_samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        stop=args.stop
    )

def get_vllm(args) -> LLM:
    config = get_hf_config()
    return LLM(model=config.model_dir,
               enable_lora=args.lora,
               max_model_len=args.max_tokens)

def get_vllm_models_with_sampling_param(args) -> Tuple[LLM, SamplingParams]:
    sampling_params = get_vllm_sampling_param(args)
    vllm = get_vllm(args)
    return vllm, sampling_params

def get_hf_models(args, lora_adapter_name="lora_adapter"):
    config = get_hf_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_dir,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, lora_config, adapter_name=lora_adapter_name)
    return model


def get_hf_tokenizer(args):
    config = get_hf_config()
    return AutoTokenizer.from_pretrained(
        config.model_dir
    )

def get_ppo_policy_model_pair(args):
    if args.lora:
        print("trying lora")
        policy_lora = get_hf_models(args, lora_adapter_name="new_policy")

        policy_lora.add_adapter("old_policy", policy_lora.peft_config["new_policy"])
        policy = LoRAWrapper(policy_lora, "new_policy")
        old_policy = LoRAWrapper(policy_lora, "old_policy")
    else:
        policy = get_hf_models(args)
        old_policy = get_hf_models(args)
    return policy, old_policy

def init_single_gpu_vllm_models_for_sft(args, device: str, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    config = get_hf_config()
    vllm_set_random_seed(args.seed)
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=config.model_dir,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    '''
    a little bit hacky, may not be used in future vllm versions
    '''
    try:
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except:
        raise NotImplementedError("module not found, your vllm versions should be 0.9.1")

    policy_device = next(policy.parameters()).device
    llm_device = llm_model.device
    if policy_device.type != "cpu" and policy_device != llm_device:
        policy_copy = policy.to(llm_device)
        state_dict = policy_copy.state_dict()
    else:
        policy_copy = None

    llm_model.load_weights(state_dict.items())
    if not policy_copy is None:
        del policy_copy
        torch.cuda.empty_cache()

def update_vllm_model_for_rft(
    args,
    policy: PreTrainedModel | OrderedDict,
    vllm: LLM,
    lora_adapter_name: None | str = None,
    lora_int_id: None | int = None
) -> Union[LoRARequest, None]:  
    lora_request = None
    if args.lora:
        assert lora_adapter_name is not None and lora_int_id is not None
        lora_path = Path(args.log_save_path) / "cache" 
        os.makedirs(lora_path, exist_ok=True)
        policy.peft_model.save_pretrained(lora_path, selected_adapters=[lora_adapter_name])
        
        lora_adapter_path = lora_path / lora_adapter_name
        if not os.path.exists(lora_adapter_path / "tokenizer.json"):
            os.symlink(Path(get_hf_config().model_dir) / "tokenizer.json", lora_adapter_path / "tokenizer.json")

        lora_request = LoRARequest(
            lora_name=lora_adapter_name,
            lora_local_path=str(lora_adapter_path),
            lora_int_id=lora_int_id)
    else:
        load_policy_into_vllm_instance(policy, vllm)
    return lora_request

def update_old_policy(
    args,
    policy: PreTrainedModel,
    old_policy: PreTrainedModel,
):
    if args.lora:
        policy_state = get_peft_model_state_dict(policy, adapter_name="new_policy")
        set_peft_model_state_dict(old_policy, policy_state, adapter_name="old_policy")
    else:
        state_dict = policy.state_dict()
        old_policy.load_state_dict(state_dict)

class LoRAWrapper:
    def __init__(self, peft_model: PeftModel, adapter_name: str):
        self.peft_model = peft_model
        self.adapter_name = adapter_name

    @contextmanager
    def as_active_adapter(self):
        original_adapter = self.peft_model.active_adapter
        try:
            self.peft_model.set_adapter(self.adapter_name)
            yield
        finally:
            self.peft_model.set_adapter(original_adapter)

    def __getattr__(self, name: str):
        attr = getattr(self.peft_model, name)

        if callable(attr):
            def wrapper(*args, **kwargs):
                with self.as_active_adapter():
                    return attr(*args, **kwargs)
            return wrapper
        else:
            return attr
    
    def __call__(self, *args, **kwargs):
        with self.as_active_adapter():
            return self.peft_model(*args, **kwargs)

        