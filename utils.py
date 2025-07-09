from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import os

def save_hf_model(args, model, tokenizer, exp_name="SFT", adapter_names: None | list[str] = None):
    if args.lora:
        assert not adapter_names is None
        model = model.merge_and_unload(adapter_names=adapter_names)
    ckpt_save_dir = Path(args.ckpt_save_dir)
    folder_name = f"{args.model}_{args.dataset}_{exp_name}_step_{args.training_step}"
    save_directory = os.path.join(args.ckpt_save_dir, folder_name)
    os.makedirs(save_directory, exist_ok=True)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

def row_to_col_oriented(dataloader: DataLoader) -> dict[str, list[str]]:
    test_data = defaultdict(list)
    for d in dataloader:
        for k, v in d.items():
            test_data[k].extend(v)
    return test_data