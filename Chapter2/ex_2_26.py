""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

mse = nn.MSELoss() # 初始化平方损失函数模块
t1 = torch.randn(5, requires_grad=True) # 随机生成张量t1
t2 = torch.randn(5, requires_grad=True) # 随机生成张量t2
mse(t1, t2) # 计算张量t1和t2之间的平方损失函数
t1 = torch.randn(5, requires_grad=True) # 随机生成张量t1
t1s = torch.sigmoid(t1)
t2 = torch.randint(0, 2, (5, )).float() # 随机生成0，1的整数序列，并转换为浮点数
bce(t1s, t2) # 计算二分类的交叉熵
bce_logits = nn.BCEWithLogitsLoss() # 使用交叉熵对数损失函数
bce_logits(t1, t2) # 计算二分类的交叉熵，可以发现和前面的结果一致
N=10 # 定义分类数目
t1 = torch.randn(5, N, requires_grad=True) # 随机产生预测张量
t2 = torch.randint(0, N, (5, )) # 随机产生目标张量
t1s = torch.nn.functional.log_softmax(t1, -1) # 计算预测张量的LogSoftmax
nll = nn.NLLLoss() # 定义NLL损失函数
nll(t1s, t2) # 计算损失函数
ce = nn.CrossEntropyLoss() # 定义交叉熵损失函数
ce(t1, t2) # 计算损失函数，可以发现和NLL损失函数的结果一致