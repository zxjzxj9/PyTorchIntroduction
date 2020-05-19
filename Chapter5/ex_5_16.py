""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch
import torch.nn as nn

rnn = nn.RNNCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)

rnn = nn.LSTMCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
cx = torch.randn(3, 20)
output = []
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)

rnn = nn.GRUCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
