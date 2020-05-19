""" 该代码可以直接使用 python ex_2_27.py 命令来进行运行（需要安装scikit-learn）
    （#号及其后面内容为注释，可以忽略）
"""

from sklearn.datasets import load_boston
boston = load_boston()

lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6) # 定义优化器
data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)

for step in range(10000):
    predict = lm(data) # 输出模型预测结果
    loss = criterion(predict, target) # 输出损失函数
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad() # 清零梯度
    loss.backward() # 反向传播
    optim.step()
# 输出结果：
# Loss: 150.855
# Loss: 113.852
# …
# Loss: 98.165
# Loss: 97.907
