import torch
import torch.optim as optim
from transformers import PreTrainedModel

def get_optimizer(args, model: PreTrainedModel):
    if args.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay

        )
    