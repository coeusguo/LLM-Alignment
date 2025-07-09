from argument import parse_arg, build_hf_config
from alignment.model_utils import get_hf_models
from alignment.dataset import get_train_data_loader
from alignment.optimizer import get_optimizer
from alignment.sft_utils import get_response_log_probs, sft_microbatch_train_step
from alignment.model_utils import get_hf_tokenizer
from utils import save_hf_model
from tqdm import tqdm
from pathlib import Path
import os
import torch

num_gpus = torch.cuda.device_count()

def sft(args):
    model = get_hf_models(args)
    tokenizer = get_hf_tokenizer(args)
    model.train()
    model.to(torch.device("cuda:0"))
    train_loader = get_train_data_loader(args, tokenizer)
    optimizer = get_optimizer(args, model)

    progress_bar = tqdm(
        train_loader,
        total=args.training_step,
        desc="SFT Training")
    for idx, data in enumerate(progress_bar):
        data = {k: v.to(model.device) for k,v in data.items()}

        out = get_response_log_probs(
            model, data["input_ids"], data["labels"])
        loss, infos = sft_microbatch_train_step(
            out["log_probs"], data["response_mask"], args)
        loss.backward()

        if (idx) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        if idx > args.training_step:
            break

    print("saving model ckpts...")
    save_hf_model(save_directory)
    print("SFT done")
        
if __name__ == "__main__":
    
    assert num_gpus >= 2, "at least 2 gpus needed"

    args = parse_arg()
    build_hf_config(args)
    sft(args)