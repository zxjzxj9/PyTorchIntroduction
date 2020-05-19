""" 该代码仅为演示函数签名和所用方法，并不能实际运行
"""

class torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
# TransformerEncoder对应的forward方法定义
forward(src, mask=None, src_key_padding_mask=None)

class torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
# TransformerDecoder对应的forward方法定义
forward(tgt, memory, tgt_mask=None, memory_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None)

class torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
    num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
    custom_encoder=None, custom_decoder=None)
# Transformer对应的forward方法定义
forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
    src_key_padding_mask=None, tgt_key_padding_mask=None,
    memory_key_padding_mask=None)
