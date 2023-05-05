import torch
import torch_xmlir.core.xpu_model as xm
import random
import numpy as np
from torch import nn


def hook_forward_fn(module, input, output):
    print(f"module: {module}")
    print(f"input: {input}")
    print(f"output: {output}")


def hook_backward_fn(module, grad_input, grad_output):
    print(f"module: {module}")
    print(f"grad_output: {grad_output}")
    print(f"grad_input: {grad_input}")
    print("*" * 20)


seed = 91
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = xm.xpu_device(eager=True)
a = torch.rand([5, 5], requires_grad=True).to(device)
drop = nn.Dropout(0.5).to(device)
drop.train()
drop.register_forward_hook(hook_forward_fn)
drop.register_backward_hook(hook_backward_fn)
b = drop(a)
d = b.sum()
d.backward()
