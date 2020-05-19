""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

import numpy as np # 导入numpy包
import torch # 导入torch包
torch.tensor([1,2,3,4]) # 转换Python列表为PyTorch张量
torch.tensor([1,2,3,4]).dtype # 查看张量数据类型
torch.tensor([1,2,3,4], dtype=torch.float32) # 指定数据类型为32位浮点数
torch.tensor([1,2,3,4], dtype=torch.float32).dtype  # 查看张量数据类型
torch.tensor(range(10)) # 转换迭代器为张量
np.array([1,2,3,4]).dtype # 查看numpy数组类型
torch.tensor(np.array([1,2,3,4])) # 转换numpy数组为PyTorch张量
torch.tensor(np.array([1,2,3,4])).dtype # 转换后PyTorch张量的类型
torch.tensor([1.0, 2.0, 3.0, 4.0]).dtype # PyTorch默认浮点类型为32位单精度
torch.tensor(np.array([1.0, 2.0, 3.0, 4.0])).dtype # numpy默认浮点类型为64位双精度
torch.tensor([[1,2], [3,4,5]]) # 列表嵌套创建张量，错误：子列表大小不一致
torch.tensor([[1,2,3], [4,5,6]]) # 列表嵌套创建张量，正确：2×3的矩阵
torch.randn(3,3).to(torch.int) # 从torch.float 转换到 torch.int，也可以调用.int()方法
torch.randint(0, 5, (3,3)).to(torch.float) # 从torch.int64到torch.float，也可以调用.float()方法

