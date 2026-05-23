import torch

def final_if(spikes_bt: torch.Tensor) -> torch.Tensor:
    return spikes_bt[:, -1]

def final_mem(mem_bt: torch.Tensor) -> torch.Tensor:
    return mem_bt[:, -1]
