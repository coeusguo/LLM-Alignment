from transformers import PreTrainedModel, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Union
from .model_utils import unwrap_model, LoRAWrapper
from peft import PeftModel
import argparse

def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: AutoTokenizer,
    prompt_format_func: Callable[Union[list[str], str], str] | None = None
    ) -> dict[str, torch.Tensor]:

    """
    prompt_strs should be applied chat template before passed in
    """
    assert len(prompt_strs) == len(output_strs)
    num_samples = len(prompt_strs)
    if prompt_format_func is not None:
        prompt_strs = prompt_format_func(prompt_strs)
    tokenized_prompts = tokenizer(prompt_strs).input_ids
    tokenized_outputs = tokenizer(output_strs).input_ids

    prompt_lens = [len(tp) for tp in tokenized_prompts]
    combined_tokens = [tokenized_prompts[i] + tokenized_outputs[i] 
                            for i in range(num_samples)]
    max_len = max([len(t) for t in combined_tokens])

    input_ids = torch.full((num_samples, max_len - 1), tokenizer.pad_token_id, dtype=torch.int64)
    labels = torch.full((num_samples, max_len - 1), tokenizer.pad_token_id, dtype=torch.int64)
    mask = torch.zeros((num_samples, max_len - 1), dtype=torch.bool)

    for idx in range(num_samples):
        token_length = len(combined_tokens[idx])
        ct = torch.tensor(combined_tokens[idx], dtype=torch.int64)
        input_ids[idx, :token_length] = ct[:(max_len-1)]
        labels[idx, :(token_length - 1)] = ct[1:]
        mask[idx, (prompt_lens[idx]-1): (token_length-1)] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": mask
    }

def compute_entropy(
    logits: None | torch.Tensor = None, 
    log_probs: None | torch.Tensor = None) -> torch.Tensor:
    """
    logits: [B, N, V]
    """
    if not logits is None:
        assert logits.dim() == 3
    else:
        assert log_probs.dim() == 3

    if log_probs is None:
        log_probs = F.log_softmax(logits, dim=-1)

    return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

def reduce_masked_entropy(
    entropy: torch.Tensor, 
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    reduced_entropy = torch.sum(entropy * mask, dim=dim, keepdims=False)
    return reduced_entropy / torch.sum(mask, dim=dim, keepdims=False, dtype=entropy.dtype)

def get_response_log_probs(
    model: PreTrainedModel | LoRAWrapper,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    chunkify: None | int = 4 # chunck the logits to avoid oom
) -> dict[str, torch.Tensor]:

    def gather_cross_entropy(logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = None
        if return_token_entropy:
            entropy = compute_entropy(log_probs=log_probs)
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return log_probs, entropy

    if chunkify is not None:
        if hasattr(model, 'peft_model'): 
            # PeftModel wrapper
            hidden_states = model.model(input_ids, use_cache=False)[0]
            peft_model = model.peft_model
            with model.as_active_adapter():
                hidden_states = unwrap_model(peft_model)(input_ids, use_cache=False)[0]
        elif hasattr(model, "model"): 
            # HF PretrainedModel
            hidden_states = unwrap_model(model)(input_ids, use_cache=False)[0]
        else:
            # Raise an error if we encounter an unknown architecture
            raise NotImplementedError(f"Unsupported model architecture: {model.__class__.__name__}")
        
        seq_len = labels.size(1)
        chunkify = min(chunkify, seq_len)
        chunk_size = seq_len // chunkify
        break_point = [i * chunk_size for i in range(chunkify)] + [seq_len]
        log_probs, entropy = [], []
        for i in range(chunkify):
            h_chunk = hidden_states[:, break_point[i]: break_point[i + 1], :]
            labels_chunk = labels[:, break_point[i]: break_point[i + 1]]
            chunked_logits = model.lm_head(h_chunk)
            log_probs_chunk, entropy_chunk = gather_cross_entropy(chunked_logits, labels_chunk)
            log_probs.append(log_probs_chunk)
            entropy.append(entropy_chunk)
        log_probs = torch.concat(log_probs, dim=-1)
        if return_token_entropy:
            entropy = torch.concat(entropy, dim=-1)
        else:
            entropy = None
    else:
        logits = model(input_ids, use_cache=False).logits
        log_probs, entropy = gather_cross_entropy(logits, labels)

    out = {
        "log_probs": log_probs,
        "token_entropy": entropy
    }

    return out
    
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None
) -> torch.Tensor:

    tensor = tensor * mask
    unnormalized_loss = torch.sum(tensor, dim=dim) / normalize_constant
    return unnormalized_loss

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    args: argparse.Namespace 
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss = -masked_normalize(policy_log_probs, response_mask, args.normalize_constant, dim=-1).mean()
    loss = loss / args.accumulation_steps
    infos = {
        "loss": loss.detach().clone()
    }
    return loss, infos

if __name__ == "__main__":
    from argument import parse_arg, build_hf_config, get_hf_config
    from .dataset import load_dataset
    from .models import get_hf_models
    from torch.distributions import Categorical

    a = torch.randn(2, 1000, 100000).cuda()
    ent = compute_entropy(a)
    gt_ent = Categorical(logits=a).entropy()
    assert torch.allclose(ent, gt_ent)
    
    args = parse_arg()
    build_hf_config(args)
    dataset = load_dataset(args)
    hf_config = get_hf_config()
    contents = []
    outputs = []
    for idx, data in enumerate(dataset["train"]):
        contents.append(data["messages"][0]["content"])
        outputs.append(data["messages"][1]["content"])
        if idx > 2:
            break
    contents = [
        "Hello, what are some key features of the Qwen3 models",
        "KFC is good"
    ]
    outputs = [
        "yes yes yes",
        "what the fxxk are you taking about"]

    tokenizer = AutoTokenizer.from_pretrained(hf_config.model_dir)
    tokens = tokenize_prompt_and_output(contents, outputs, tokenizer)
    model = get_hf_models(args)
    probs = get_response_log_probs(model, tokens["input_ids"], tokens["labels"])
    # tensor = masked_normalize(probs["log_prob"], tokens["response_mask"])
    sft_microbatch_train_step(probs["log_prob"], tokens["response_mask"], args)
