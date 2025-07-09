import argparse
import os
import datetime
from pathlib import Path
from dataclasses import dataclass, field, InitVar

HF_CONFIG = None

@dataclass
class HFConfig:
    args: InitVar[argparse.Namespace]
    cache_dir: str = os.path.join("/", "data", "huggingface")
    model_dir: str | None = field(init=False)
    
    available_models = ["qwen2-0.5b", "qwen3-0.6b"]
    available_datasets = ["gsm8k", "AM-DeepSeek-R1-Distilled-1.4M"]
    dataset_dir: str | None = field(init=False)

    def _get_last_commit_folder(self, parent_dir):
        
        snapshots = os.path.join(parent_dir, "snapshots")
        if os.path.isdir(snapshots):
            last_commit_folder = os.listdir(snapshots)[-1]
            return os.path.join(snapshots, last_commit_folder)
        else:
            # no snapshot folder, i.e. current folder already contains the data
            return parent_dir

    def _get_list_by_name(self, name, child_dir_name: str="hub"):
        name = name.lower()
        child_path = os.path.join(self.cache_dir, child_dir_name)
        return os.listdir(child_path), child_path

    def __post_init__(self, args):
        model = args.model
        dataset = args.dataset

        if args.ckpt_path is not None:
            assert os.path.isdir(args.ckpt_path), f"the path ({args.ckpt_path} does not exist)"
            self.model_dir = args.ckpt_path
        else:
            models, model_root = self._get_list_by_name(args.model, "hub")
            for m in models:
                name, size = m.split("--")[-1].lower().split("-")[:2]
                if name in model and size in model:
                    self.model_dir = self._get_last_commit_folder(
                        os.path.join(model_root, m))
                    break

        assert dataset in self.available_datasets
        datasets, dataset_root = self._get_list_by_name(dataset, "datasets")
        for d in datasets:
            if dataset in d:
                self.dataset_dir = self._get_last_commit_folder(
                    os.path.join(dataset_root, d)
                )  
            

def build_hf_config(args):
    global HF_CONFIG
    if HF_CONFIG is None:
        HF_CONFIG = HFConfig(args)
    return HF_CONFIG

def get_hf_config():
    global HF_CONFIG
    assert not HF_CONFIG is None
    return HF_CONFIG

def parse_arg():
    parser = argparse.ArgumentParser(description="evluate LLMs")

    # random seed
    parser.add_argument("-seed", type=int, default=114514) 

    # model
    parser.add_argument("-model", type=str, default="qwen3-0.6b", 
                        help=f"available models at this stage: {HFConfig.available_models}")
    parser.add_argument("-ckpt-path", type=str, default=None, 
                        help=f"saved ckpt direction")

    # lora
    parser.add_argument("-lora", action="store_true")
    parser.add_argument("-lora-rank", type=int, default=8)
    parser.add_argument("-lora-alpha", type=float, default=16.0)
    parser.add_argument("-lora-target-modules", nargs="+", type=str, default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("-lora-dropout", type=float, default=0.05)

    # vllm related
    parser.add_argument("-temperature", type=float, default=0.8)
    parser.add_argument("-top-p", type=float, default=1.0)
    parser.add_argument("-max-tokens", type=int, default=1024)
    parser.add_argument("-min-tokens", type=int, default=4)
    parser.add_argument("-num-samples-per-prompt", type=int, default=1)
    parser.add_argument("-stop", nargs="*", type=str, default=[], metavar="STRING",
                        help="One or more strings to use as stop sequences. "
                             "Generation will stop when any of these strings are generated. "
                             "Example: --stop \"<|endoftext|>\" \"Observation:\"")

    # dataset related
    parser.add_argument("-dataset", type=str, default="gsm8k",
                        help=f"available datasets at this stage: {HFConfig.available_datasets}")

    # eval related
    parser.add_argument("-log_dir", type=str, default="eval_data")

    # training related
    parser.add_argument("-accumulation-steps", type=int, default=4)
    parser.add_argument("-normalize-constant", type=float, default=1.0)
    parser.add_argument("-batch-size", type=int, default=16)

    parser.add_argument("-optimizer", type=str, default="adamW")
    parser.add_argument("-lr", type=float, default=2e-5)
    parser.add_argument("-betas", type=float, nargs='+', default=[0.9, 0.95])
    parser.add_argument("-weight-decay", type=float, default=1e-2)
    parser.add_argument("-max-grad-norm", type=float, default=None, help="grad norm will no larger than max_grad_norm")

    parser.add_argument("-training-step", type=int, default=1000)
    parser.add_argument("-eval-every", type=int, default=1000)
    parser.add_argument("-ckpt-save-dir", type=str, default="./ckpts")

    # grpo(&ppo)
    parser.add_argument("-ppo-clip-range", type=float, default=0.1)
    # parser.add_argument("-grpo-group-size", type=int, default=8)
    parser.add_argument("-epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("-rollout-batch-size", type=int, default=16)
    parser.add_argument("-advantage-eps", type=float, default=1e-6)
    parser.add_argument("-normalize-by-std", action="store_true")

    # prompt
    parser.add_argument("-prompt-template", type=str, default="r1_zero")

    # logging
    parser.add_argument("-log-every", type=int, default=10)
    parser.add_argument("-log-save-path", type=str, default="log")

    args = parser.parse_args()

    # some sanity checks
    assert args.batch_size * args.num_samples_per_prompt % args.rollout_batch_size == 0, \
                    "batch-size * num-samples-per-prompt must be a muliple of roll-out-batch-size!"
    # assert args.num_samples_per_prompt % args.grpo_group_size == 0, \
    #                 "num-samples-per-prompt must be a multiple of grpo-group-size"

    os.makedirs(args.log_save_path, exist_ok=True)
    args.log_save_path = Path(args.log_save_path) / f"{args.model}_{datetime.datetime.now()}".replace(" ", "-")
    
    
    return args
