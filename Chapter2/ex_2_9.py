""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import torch

t = torch.randn(3,4,5) # 产生一个3×4×5的张量
t.ndimension() # 获取维度的数目
t.nelement() # 获取该张量的总元素数目
t.size() # 获取该张量每个维度的大小，调用方法
t.shape # 获取该张量每个维度的大小，访问属性
t.size(0) # 获取该张量维度0的大小，调用方法
t = torch.randn(12) # 产生大小为12的向量
t.view(3, 4) # 向量改变形状为3×4的矩阵
t.view(4, 3) # 向量改变形状为4×3的矩阵
t.view(-1, 4) # 第一个维度为-1，PyTorch会自动计算该维度的具体值
t # view方法不改变底层数据，改变view后张量会改变原来的张量
t.view(4, 3)[0, 0] = 1.0
t.data_ptr() # 获取张量的数据指针
t.view(3,4).data_ptr() # 数据指针不改变
t.view(4,3).data_ptr() # 同上，不改变
t.view(3,4).contiguous().data_ptr() # 同上，不改变
t.view(4,3).contiguous().data_ptr() # 同上，不改变
t.view(3,4).transpose(0,1).data_ptr() # transpose方法交换两个维度的步长
t.view(3,4).transpose(0,1).contiguous().data_ptr() # 步长和维度不兼容，重新生成张量
