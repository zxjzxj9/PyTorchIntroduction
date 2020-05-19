""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t1 = torch.randn(3, 3, requires_grad=True) # 初始化t1张量
t2 = t1.pow(2).sum() # 根据t1张量计算t2张量
torch.autograd.grad(t2, t1) # t2张量对t1张量求导