""" 该代码可以直接使用 python ex_2_37.py 运行（需要安装TensorBoard）
    （#号及其后面内容为注释，可以忽略）
    结合ex_2_20.py中定义的LinearModel模型
"""


from sklearn.datasets import load_boston
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

from ex_2_20 import LinearModel

boston = load_boston()
lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6)
data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)
writer = SummaryWriter() # 定义TensorBoard输出类
for step in range(10000):
    predict = lm(data)
    loss = criterion(predict, target)
    writer.add_scalar("Loss/train", loss, step) # 输出损失函数
    writer.add_histogram("Param/weight", lm.weight, step) # 输出权重直方图
    writer.add_histogram("Param/bias", lm.bias, step) # 输出偏置直方图
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad()
    loss.backward()
    optim.step()
