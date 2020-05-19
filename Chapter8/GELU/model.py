# 静态加载
import torch
import gelu

# 同样可以通过 gelu = GELU.apply使用这个激活函数
class GELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return gelu.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        return gelu.backward(grad_output, input)

# 动态加载
import torch
from torch.utils.cpp_extension import load

# PyTorch会进行自动编译，生成对应的模块
gelu = load(name="gelu", sources=["gelu/gelu.cc"])

class GELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return gelu.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        return gelu.backward(grad_output, input)
