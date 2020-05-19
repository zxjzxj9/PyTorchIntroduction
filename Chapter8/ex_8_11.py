""" 本代码仅作为钩子函数的演示代码
"""

# 模块执行之前的前向计算钩子的定义
# 定义nn.Module的一个实例模块
module = ...
def hook(module, input):
    # 对模块权重或者输入进行操作的代码
    # 函数结果可以返回修改后的张量或者None
    return input
handle = module.register_forward_pre_hook(hook)

# 模块执行之后的前向计算钩子的定义
# 定义nn.Module的一个实例模块
module = ...
def hook(module, input, output):
    # 对模块权重或者输入/输出进行操作的代码
    # 函数结果可以返回修改后的张量或者None
    return output
handle = module.register_forward_hook(hook)

# 模块执行之后的反向传播钩子的定义
# 定义nn.Module的一个实例模块
module = ...
def hook(module, grad_input, grad_output):
    # 对模块权重或者输入/输出梯度进行操作的代码
    # 函数结果可以返回修改后的张量或者None
    return output
handle = module.register_backward_hook(hook)

# 钩子的使用方法示例
import torch
import torch.nn as nn
def print_pre_shape(module, input):
    print("模块前钩子")
    print(module.weight.shape)
    print(input[0].shape)
def print_post_shape(module, input, output):
    print("模块后钩子")
    print(module.weight.shape)
    print(input[0].shape)
    print(output[0].shape)
def print_grad_shape(module, grad_input, grad_output):
    print("梯度钩子")
    print(module.weight.grad.shape)
    print(grad_input[0].shape)
    print(grad_output[0].shape)
conv = nn.Conv2d(16, 32, kernel_size=(3,3))
handle1 = conv.register_forward_pre_hook(print_pre_shape)
handle2 = conv.register_forward_hook(print_post_shape)
handle3 = conv.register_backward_hook(print_grad_shape)
input = torch.randn(4, 16, 128, 128, requires_grad=True)
ret = conv(input)