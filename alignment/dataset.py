from datasets import load_dataset as hf_load_dataset
from argument import get_hf_config
from torch.utils.data import DataLoader
from functools import partial
from .prompt_templates import format_prompt
from .sft_utils import tokenize_prompt_and_output
from transformers import AutoTokenizer
from typing import Callable

def load_dataset(args):
    dataset = {}
    hf_config = get_hf_config()
    preprocess_func = identity_preprocessor

    if args.dataset == "gsm8k":
        dataset.update(load_gms8k_dataset(hf_config))
        # preprocess_func = gsm8k_preprocessor

    elif args.dataset == "AM-DeepSeek-R1-Distilled-1.4M":
        dataset.update(load_am_r1_distilled_dataset(hf_config))
    else:
        raise NotImplementedError(f"{args.dataset} not implemented")
    
    return preprocess_func(dataset)

def load_gms8k_dataset(hf_config):
    return hf_load_dataset(hf_config.dataset_dir, "main")

def load_am_r1_distilled_dataset(hf_config):
    """
    train
        -messages
            0 - role: user, content: ....
            1 - role: assistant, content: ...
    """
    return hf_load_dataset(hf_config.dataset_dir, "default", streaming=True)

def identity_preprocessor(data):
    return data

def gsm8k_r1_zero_template_preprocessor(data):
    split_mark = "####"
    for d in data:
        assert split_mark in d["answer"], f"{split_mark} not found in {d['answer']}"
        rationale, answer = d["answer"].split(split_mark)
        answer = answer.strip()
        d["answer"] = f"{rationale}</think>\n<answer>{answer}</answer>"
    return data

def gsm8k_sft_collator(
    tokenizer: None | AutoTokenizer = None, 
    format_prompt: None | Callable[[list[str]], list[str]] = None,
    format_type: str = "r1_zero",
    pre_processor: Callable[[list[str]], list[str]] = gsm8k_r1_zero_template_preprocessor):
    def wrapper(samples):
        samples = pre_processor(samples)
        prompt = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        if tokenizer is not None:
            return tokenize_prompt_and_output(prompt, answers, tokenizer, format_prompt)
        out = {
            "questions": prompt,
            "answers": answers}
        if format_prompt is not None:
            out["questions"] = format_prompt(out["questions"])
        return out
    return wrapper

def gsm8k_sft_data_loader(args, tokenizer, domain="train"):
    dataset = load_dataset(args)
    format_prompt_fn = partial(format_prompt, format=args.prompt_template)
    return DataLoader(
        dataset[domain], 
        batch_size=args.batch_size,
        collate_fn=gsm8k_sft_collator(tokenizer, format_prompt_fn, args.prompt_template))

def get_train_data_loader(args, tokenizer):
    if args.dataset == "gsm8k":
        return gsm8k_sft_data_loader(args, tokenizer, domain="train")
    else:
        raise NotImplementedError(f"{args.dataset} train loader not implemented")

def get_test_data_loader(args, tokenizer):
    if args.dataset == "gsm8k":
        return gsm8k_sft_data_loader(args, tokenizer, domain="test")
    else:
        raise NotImplementedError(f"{args.dataset} train loader not implemented")