""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t1 = torch.randn(3,4,5) # 定义3×4×5的张量1
t2 = torch.randn(3,5) # 定义 3×5的张量2
t1
t2
t2 = t2.unsqueeze(1) # 张量2的形状变为3×1×5
t1.shape
t2.shape
t3 = t1 + t2 # 广播求和，最后结果为3×4×5的张量
t3
t3.shape