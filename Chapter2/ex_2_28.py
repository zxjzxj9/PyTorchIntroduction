""" 该代码为示例代码，演示了优化器如何使用
"""

import torch
optim = torch.optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)