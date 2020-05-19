""" 以下代码仅作为WaveNet的实现参考
"""

import torch
import torch.nn as nn

# 因果卷积模块
class CausalConv(nn.Module):

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 local_channels, dropout=0.05, dilation=1, bias=True):

        super(CausalConv, self).__init__()
        self.dropout = dropout

        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(residual_channels, gate_channels, 
                              kernel_size, padding=padding,
                              dilation=dilation, bias=bias)

        self.conv1x1_local = Conv1d1x1(local_channels,
                                       gate_channels, bias=False)
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, 
                                     residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels,
                                      residual_channels, bias=bias)

    def forward(self, x, x_local):

        # x为音频信号，x_local为梅尔过滤器特征上采样到和x维度相同后的结果
        # 假设输入x的大小为N×C×T，其中N为批次大小，C为输入特征大小，
        # T为序列长度
        # x_local大小和x大小相同

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 因果卷积
        x = self.conv(x)
        x = x[:, :, :residual.size(-1)]

        # 因果卷积结果分割
        a, b = x.split(x.size(-1) // 2, dim=-1)
        # 加入局域特征的调制
        c = self.conv1x1_local(x_local)
        ca, cb = c.split(c.size(-1) // 2, dim=-1)
        a, b = a + ca, b + cb

        x = torch.tanh(a) * torch.sigmoid(b)

        s = self.conv1x1_skip(x)
        x = self.conv1x1_out(x)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

# WaveNet模型代码
class WaveNet(nn.Module):

    def __init__(self, out_channels=256, layers=20,
                 layers_per_stack = 2,
                 residual_channels=512,
                 gate_channels=512,
                 mel_channels = 80,
                 mel_kernel = 1024,
                 mel_stride = 256,
                 skip_out_channels=512,
                 kernel_size=3, dropout= 0.05,
                 local_channels=512):

        super(WaveNet, self).__init__()

        self.out_channels = out_channels
        self.local_channels = local_channels
        self.first_conv = nn.Conv1d(out_channels, residual_channels, 1)

        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = CausalConv(residual_channels, gate_channels, kernel_size,
                              local_channels, dropout, dilation, True)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_out_channels, skip_out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_out_channels, out_channels, 1),
        ])

        self.upsample_net = nn.ConvTranspose1d(mel_channels, gate_channels, 
                                               mel_kernel, mel_stride)

    def forward(self, x, x_local):

        # x为音频信号，x_local为梅尔过滤器特征
        B, _, T = x.size()
        # 对特征进行上采样，输出和音频信号长度相同的信号
        c = self.upsample_net(x_local)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        # 输出每个强度的概率
        x = F.softmax(x, dim=1)
        return x
