""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

a = torch.randn(2,3,4) # 随机产生张量
b = torch.randn(2,4,3)
a.bmm(b) # 批次矩阵乘法的结果
torch.einsum("bnk,bkl->bnl", a, b) # einsum函数的结果，和前面的结果一致
