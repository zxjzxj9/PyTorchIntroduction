""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
    本代码仅为示例代码，需要添加模型实现细节才能运行
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, ...): # 定义类的初始化函数，...是用户的传入参数
        super(Model, self).__init__()
        ... # 根据传入的参数来定义子模块
    
    def forward(self, ...): # 定义前向计算的输入参数，...一般是张量或者其他的参数
        ret = ... # 根据传入的张量和子模块计算返回张量
        return ret
