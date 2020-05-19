"""  本代码定义的模块可以用于其它的Python脚本，已经Python的交互命令行界面
    （#号及其后面内容为注释，可以忽略）
"""

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim

        self.weight = nn.Parameter(torch.randn(ndim, 1)) # 定义权重
        self.bias = nn.Parameter(torch.randn(1)) # 定义偏置

    def forward(self, x):
        # 定义线性模型 y = Wx + b
        return x.mm(self.weight) + self.bias