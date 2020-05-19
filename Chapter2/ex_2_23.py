""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t1 = torch.randn(3, 3, requires_grad=True) # 定义一个3×3的张量
t1
t2 = t1.pow(2).sum() # 计算张量的所有分量平方和
t2.backward() # 反向传播
t1.grad # 梯度是张量原始分量的2倍
t2 = t1.pow(2).sum() # 再次计算所有分量的平方和
t2.backward() # 再次反向传播
t1.grad # 梯度累积
t1.grad.zero_() # 单个张量清零梯度的方法