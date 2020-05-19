""" 该代码定义了LeNet模型的训练过程
"""

import torch
import torch.nn as nn
from model import LeNet

# ... 此处略去定义训练数据载入器的代码，具体可参考代码4.3

model = LeNet() # 定义LeNet模型
model.train() # 切换模型到训练状态
lr = 0.01 # 定义学习率
criterion = nn.CrossEntropyLoss() # 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
    weight_decay=5e-4) # 定义随机梯度下降优化器

train_loss = 0
correct = 0
total = 0

for batch_idx, (inputs, targets) in enumerate(data_train_loader):

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
