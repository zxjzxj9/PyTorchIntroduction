""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
    结合ex_2_20.py中定义的LinearModel模型
"""

import torch
import torch.nn as nn
from ex_2_20 import LinearModel

lm = LinearModel(5) # 定义线性模型
lm.state_dict() # 获取状态字典
t = lm.state_dict() # 保存状态字典
lm = LinearModel(5) # 重新定义线性模型
lm.state_dict() # 新的状态字典，模型参数和原来的不同
lm.load_state_dict(t) # 载入原来的状态字典
lm.state_dict() # 模型参数已更新