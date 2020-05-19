""" 该代码仅为演示函数签名所用，并不能实际运行
"""

# 最大池化模块
class torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1,return_indices=False, ceil_mode=False)
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,return_indices=False, ceil_mode=False)
class torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1,return_indices=False, ceil_mode=False)

# 平均池化模块
class torch.nn.AvgPool1d(kernel_size, stride=None, padding=0,
    ceil_mode=False, count_include_pad=True)
class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0,
    ceil_mode=False, count_include_pad=True, divisor_override=None)
class torch.nn.AvgPool3d(kernel_size, stride=None, padding=0,
    ceil_mode=False, count_include_pad=True, divisor_override=None)

# 乘幂平均池化模块
class torch.nn.LPPool1d(norm_type, kernel_size, stride=None,
    ceil_mode=False)
class torch.nn.LPPool2d(norm_type, kernel_size, stride=None,
    ceil_mode=False)

# 分数最大池化模块
class torch.nn.FractionalMaxPool2d(kernel_size, output_size=None,
    output_ratio=None, return_indices=False, _random_samples=None)
class torch.nn.FractionalMaxPool3d(kernel_size, output_size=None,
    output_ratio=None, return_indices=False, _random_samples=None)

# 自适应池化模块
class torch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)
class torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
class torch.nn.AdaptiveMaxPool3d(output_size, return_indices=False)
class torch.nn.AdaptiveAvgPool1d(output_size)
class torch.nn.AdaptiveAvgPool2d(output_size)
class torch.nn.AdaptiveAvgPool3d(output_size)

# 最大反池化模块
class torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
class torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
class torch.nn.MaxUnpool3d(kernel_size, stride=None, padding=0)
