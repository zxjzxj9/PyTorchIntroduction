""" 本文件中的代码可以通过使用命令 python ex_3_18.py 运行  
   （#号及其后面内容为注释，可以忽略）
"""

import torch.nn as nn

# 模块列表的使用方法
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # 模块列表的迭代和使用方法与Python的普通列表一致
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

# 模块字典的使用方法
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
