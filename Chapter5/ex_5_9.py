""" 该代码仅为演示函数签名和所用方法，并不能实际运行
"""

torch.nn.utils.rnn.pack_padded_sequence(input, lengths, 
    batch_first=False, enforce_sorted=True)
torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False,
    padding_value=0.0, total_length=None)