import torch

lib = torch.library.Library("accinte", "IMPL", "Meta")


@torch.library.impl(lib, "custom_abs")
def custom_abs(self):
    return torch.empty_like(self)
