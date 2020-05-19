""" 该代码仅为演示函数签名所用，并不能实际运行
"""

# 批次归一化
class torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

# 组归一化
class torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)

# 实例归一化
class torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
class torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
class torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

# 层归一化
class torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)

# 局部响应归一化
class torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)