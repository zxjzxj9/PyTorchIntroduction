""" 该代码仅为演示函数签名和所用方法，并不能实际运行
"""

class torch.nn.MultiheadAttention(embed_dim, num_heads,
    dropout=0.0, bias=True, add_bias_kv=False,
    add_zero_attn=False, kdim=None, vdim=None)

# 对应的forward方法定义
forward(query, key, value, key_padding_mask=None,
    need_weights=True, attn_mask=None)
