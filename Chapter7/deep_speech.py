""" 本代码仅作为DeepSpeech模型的实现参考
class BNGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BNGRU, self).__init__()

        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)

    def forward(self, x, xlen):
        maxlen = x.size(2)
        x = self.bn(x)
        # N×C×T -> T×N×C
        x = x.permute(2, 0, 1)
        x = nn.utils.rnn.pack_padded_sequence(x, xlen) 
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=maxlen)
        x = x[..., :self.hidden_size] + x[..., self.hidden_size:]
        # T×N×C -> N×C×T
        x = x.permute(1, 2, 0)
        return x

class DeepSpeech(nn.Module):

    def __init__(self, mel_channel, channels, kernel_dims, strides, 
        num_layers, hidden_size, char_size):

        super(DeepSpeech, self).__init__()
        self.kernel_dims = kernel_dims
        self.strides = strides
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.char_size = char_size

        self.cnns = nn.ModuleList()
        in_channel = mel_channel
        for c, k, s in zip(channels, kernel_dims, strides):
            self.cnns.append(nn.Conv1d(in_channel, c, k, 
                stride=s, padding=c//2))
            self.cnns.append(nn.BatchNorm1d(c))
            self.cnns.append(nn.ReLU(inplace=True))
            in_channel = c
        self.cnns = nn.Sequential(*self.cnns)        

        self.rnns = nn.ModuleList()
        for _ in range(num_layers):
            self.rnns.append(BNGRU(in_channel, hidden_size))
            in_channel = hidden_size

        self.norm = nn.BatchNorm1d(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, char_size)
        ) 

    def forward(self, x, xlen):
        # T×N×C -> N×C×T
        x = x.permute(1, 2, 0)
        x = self.cnns(x)

        for rnn in self.rnns:
            x = rnn(x, xlen)
        x = self.norm(x)

        # N×C×T -> T×N×C
        x = x.permute(2, 0, 1)
        x = self.proj(x)

        return F.log_softmax(x, -1)
"""

import torch
import torch.nn as nn

class BNGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BNGRU, self).__init__()

        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)

    def forward(self, x, xlen):
        maxlen = x.size(2)
        x = self.bn(x)
        # N×C×T -> T×N×C
        x = x.permute(2, 0, 1)
        x = nn.utils.rnn.pack_padded_sequence(x, xlen) 
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=maxlen)
        x = x[..., :self.hidden_size] + x[..., self.hidden_size:]
        # T×N×C -> N×C×T
        x = x.permute(1, 2, 0)
        return x

class DeepSpeech(nn.Module):

    def __init__(self, mel_channel, channels, kernel_dims, strides, 
        num_layers, hidden_size, char_size):

        super(DeepSpeech, self).__init__()
        self.kernel_dims = kernel_dims
        self.strides = strides
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.char_size = char_size

        self.cnns = nn.ModuleList()
        in_channel = mel_channel
        for c, k, s in zip(channels, kernel_dims, strides):
            self.cnns.append(nn.Conv1d(in_channel, c, k, 
                stride=s, padding=c//2))
            self.cnns.append(nn.BatchNorm1d(c))
            self.cnns.append(nn.ReLU(inplace=True))
            in_channel = c
        self.cnns = nn.Sequential(*self.cnns)        

        self.rnns = nn.ModuleList()
        for _ in range(num_layers):
            self.rnns.append(BNGRU(in_channel, hidden_size))
            in_channel = hidden_size

        self.norm = nn.BatchNorm1d(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, char_size)
        ) 

    def forward(self, x, xlen):
        # T×N×C -> N×C×T
        x = x.permute(1, 2, 0)
        x = self.cnns(x)

        for rnn in self.rnns:
            x = rnn(x, xlen)
        x = self.norm(x)

        # N×C×T -> T×N×C
        x = x.permute(2, 0, 1)
        x = self.proj(x)

        return F.log_softmax(x, -1)
