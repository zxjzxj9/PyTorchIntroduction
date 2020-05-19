""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
    结合ex_2_20.py中定义的LinearModel模型
"""

import torch
import torch.nn as nn
from ex_2_20 import LinearModel

lm = LinearModel(5) # 定义线性回归模型，特征数为5
x = torch.randn(4, 5) # 定义随机输入，迷你批次大小为4
lm(x) # 得到每个迷你批次的输出