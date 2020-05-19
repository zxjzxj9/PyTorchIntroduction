""" 该代码仅为演示函数签名和使用方法
"""

# 函数签名 
class torch.nn.CosineSimilarity(dim=1, eps=1e-08)

# 使用方法
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)