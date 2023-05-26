from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
import torch_xmlir
import torch.cuda.amp.autocast_mode

class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, weight:torch.Tensor, bias:torch.Tensor=None):
        ctx.save_for_backward(input, weight, bias)
        # if bias is None:
        #     bias = torch.zeros(weight.shape[0], dtype=torch.float32, device=weight.device)
        # output = torch.empty(input.shape[0], weight.shape[0], dtype=input.dtype, device=input.device)
        # torch.ops.custom_ops.fc_fusion(input, weight, output, bias, w_trans=True)
        return torch.ops.custom_ops.fc_with_trans_bias(input, weight, bias, w_trans=True)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        # print(input)
        # print(input.shape)
        # print(weight)
        # print(weight.shape)
        # print(ctx.needs_input_grad)
        # temp = torch.empty_like(weight)
        # print(temp)
        # print(temp.shape)
        # exit()
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = torch.empty_like(input)
            # dump_bias = torch.zeros(input.shape[1], dtype=input.dtype, device=input.device)
            # torch.ops.custom_ops.fc_fusion(grad_output, weight, grad_input, dump_bias)
            grad_input = torch.ops.custom_ops.fc_with_trans_bias(grad_output, weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = torch.empty_like(weight)
            # dump_bias = torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)
            # torch.ops.custom_ops.fc_fusion(grad_output, input, grad_weight, dump_bias, x_trans=True)
            grad_weight = torch.ops.custom_ops.fc_with_trans_bias(grad_output, input, x_trans=True)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

class CustomLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(CustomLinear, self).__init__(in_features, out_features, bias, device, dtype)
    def forward(self, input: Tensor) -> Tensor:
        if input.dtype == torch.float16:
            weight = self.weight.to(torch.float16)
        elif torch_xmlir._XMLIRC.is_autocast_enabled():
            input = input.to(torch.float16)
            weight = self.weight.to(torch.float16)
        else:
            weight = self.weight
        output = CustomLinearFunction.apply(input.view(-1, input.shape[-1]), weight, self.bias)
        return output.view(input.shape[:-1] + (self.out_features,))