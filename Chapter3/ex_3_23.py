""" 该代码仅为演示函数签名所用，并不能实际运行
"""
# 计算增益系数函数签名
torch.nn.init.calculate_gain(nonlinearity, param=None)

# 计算并且打印增益系数
gain = nn.init.calculate_gain('leaky_relu', 0.2)
print(gain)

# 参数初始化函数签名
torch.nn.init.uniform_(tensor, a=0.0, b=1.0)
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
torch.nn.init.ones_(tensor)
torch.nn.init.zeros_(tensor)
torch.nn.init.xavier_uniform_(tensor, gain=1.0)
torch.nn.init.xavier_normal_(tensor, gain=1.0)
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in',
    nonlinearity='leaky_relu')
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in',
    nonlinearity='leaky_relu')
