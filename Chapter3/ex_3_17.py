""" 本文件中的代码可以通过使用命令 python ex_3_17.py 运行  
   （#号及其后面内容为注释，可以忽略）
"""

import torch.nn as nn

# 情况1. 使用参数来构建顺序模型
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

print(model)

# 情况2. 使用顺序字典来构建顺序模型
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

print(model)