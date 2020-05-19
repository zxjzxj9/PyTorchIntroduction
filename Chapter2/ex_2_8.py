""" 为了能够现实下列代码的执行效果，请在安装PyTorch之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
    注意：本文件中的一些代码要求系统安装有GPU，有些代码可能要求系统安装多个GPU
"""

import torch # 导入torch包

torch.randn(3, 3, device="cpu") # 获取存储在CPU上的一个张量
torch.randn(3, 3, device="cuda:0") # 获取存储在0号GPU上的一个张量
torch.randn(3, 3, device="cuda:1") # 获取存储在1号GPU上的一个张量
torch.randn(3, 3, device="cuda:1").device # 获取当前张量的设备
torch.randn(3, 3, device="cuda:1").cpu().device # 张量从1号GPU转移到CPU
torch.randn(3, 3, device="cuda:1").cuda(1).device # 张量保持设备不变
torch.randn(3, 3, device="cuda:1").cuda(0).device # 张量从1号GPU转移到0号GPU
torch.randn(3, 3, device="cuda:1").to("cuda:0").device # 张量从1号GPU转移到0号GPU