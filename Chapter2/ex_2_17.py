""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t = torch.rand(3, 4) # 随机生成一个张量
t.shape
t.unsqueeze(-1).shape # 扩增最后一个维度
t.unsqueeze(-1).unsqueeze(-1).shape # 继续扩增最后一个维度
t = torch.rand(1,3,4,1) # 随机生成一个张量，有两个维度大小为1
t.shape
t.squeeze().shape # 压缩所有大小为1的维度