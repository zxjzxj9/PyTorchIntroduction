""" 该代码仅为演示函数签名和所用方法，并不能实际运行
"""

class torch.nn.TransformerEncoderLayer(d_model,
    nhead, dim_feedforward=2048, dropout=0.1)
# TransformerEncoderLayer对应的forward方法定义
forward(src, src_mask=None, src_key_padding_mask=None)

class torch.nn.TransformerDecoderLayer(d_model,
    nhead, dim_feedforward=2048, dropout=0.1)
# TransformerDecoderLayer对应的forward方法定义
forward(tgt, memory, tgt_mask=None, memory_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None)
