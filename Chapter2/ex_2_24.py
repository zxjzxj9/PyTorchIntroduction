""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t1 = torch.randn(3, 3, requires_grad=True) # 初始化t1张量
t2 = t1.sum()
t2 # t2的计算构建了计算图，输出结果带有grad_fn
with torch.no_grad():
    t3 = t1.sum()
t3 # t3的计算没有构建计算图，输出结果没有grad_fn
t1.sum() # 保持原来的计算图
t1.sum().detach() # 和原来的计算图分离