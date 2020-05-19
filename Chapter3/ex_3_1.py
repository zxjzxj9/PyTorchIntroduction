""" 该代码仅为演示函数签名所用，并不能实际运行
"""

# 类签名
torch.nn.Linear(in_features, out_features, bias=True)

# 类使用方法
import torch.nn as nn
ndim = ... # 定义输入的特征维数（整数）
lm = nn.Linear(ndim, 1)

# 类使用示例
import torch
import torch.nn as nn
lm = nn.Linear(5, 10) # 输入特征5，输出特征10
t = torch.randn(4, 5) # 迷你批次大小4，特征大小5
lm(t).shape # 迷你批次大小4，特征大小10
