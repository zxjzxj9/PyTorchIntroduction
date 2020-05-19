""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch
import torch.nn as nn

embedding = nn.Embedding(10, 4)
embedding.weight
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)
embedding = nn.Embedding(10, 4, padding_idx=0) # 定义10×4的词嵌入张量，其中索引为0的词向量为0
embedding.weight
input = torch.LongTensor([[0,2,0,5]])
embedding(input)