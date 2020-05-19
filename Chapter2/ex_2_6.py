""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
    这段代码合并了代码2.6和代码2.7的内容
"""
import numpy as np # 导入numpy包
import torch # 导入torch包

t = torch.randn(3,3) # 生成一个随机正态分布的张量t
print(t)
torch.zeros_like(t) # 生成一个元素全为0的张量，形状和给定张量t相同
torch.ones_like(t) # 生成一个元素全为1的张量，形状和给定张量t相同
torch.rand_like(t) # 生成一个元素服从[0, 1)上的均匀分布的张量，形状和给定张量t相同
torch.randn_like(t) # 生成一个元素服从标准正态分布的张量，形状和给定张量t相同

t.new_tensor([1,2,3]).dtype # 根据Python列表生成张量，注意这里输出的是单精度浮点数
t.new_zeros(3, 3) # 生成相同类型且元素全为0的张量
t.new_ones(3,3) # 生成相同类型且元素全为1的张量