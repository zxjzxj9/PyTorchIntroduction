""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
    结合ex_2_20.py中定义的LinearModel模型
"""

import torch
import torch.nn as nn
from ex_2_20 import LinearModel

lm = LinearModel(5) # 定义线性模型
x = torch.randn(4, 5) # 定义模型输入
lm(x) # 根据模型获取输入对应的输出
lm.named_parameters() # 获取模型参数（带名字）的生成器
list(lm.named_parameters()) # 转换生成器为列表
lm.parameters() # 获取模型参数（不带名字）的生成器
list(lm.parameters()) # 转换生成器为列表
lm.cuda() # 将模型参数移到GPU上
list(lm.parameters()) # 显示模型参数，可以看到已经移到了GPU上（device='cuda:0'）
lm.half() # 转换模型参数为半精度浮点数
list(lm.parameters()) # 显示模型参数，可以看到已经转换为了半精度浮点数（dtype=torch.float16）
