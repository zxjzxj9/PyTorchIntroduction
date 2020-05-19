""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t1 = torch.rand(2, 3)
t2 = torch.rand(2, 3)
t1.add(t2) # 四则运算，不改变参与运算的张量的值
t1+t2
t1.sub(t2)
t1-t2
t1.mul(t2)
t1*t2
t1.div(t2)
t1/t2
t1
t1.add_(t2) # 四则运算，改变参与运算张量的值
t1