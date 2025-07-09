import torch

def print_gpu_memory(note=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(1) / 1e9
        reserved = torch.cuda.memory_reserved(1) / 1e9
        print(f"GPU Memory {note}: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
