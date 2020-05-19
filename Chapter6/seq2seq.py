""" 本代码仅作为Seq2Seq模型的实现参考
"""

import torch
import torch.nn as nn

# 编码器
class LSTMEncoder(nn.Module):
    def __init__(
        self, dictionary,embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        padding_value=0.):

        super(LSTMEncoder, self).__init__()
        self.dictionary = dictionary
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        # 获取单词表中'<PAD>'单词对应的整数索引
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim,
                                      self.padding_idx)

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        # 假设输入为 L×N，其中L为最大序列长度，N为迷你批次大小
        seqlen, bsz = src_tokens.size()
        # 查找输入序列对应的词嵌入
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # 对输入张量进行打包
        packed_x = nn.utils.rnn.pack_padded_sequence(x,
                                                     src_lengths.data.tolist())
        # 获取隐含状态大小
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        # 初始化隐含状态为全零张量
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        # 根据输入张量计算LSTM输出
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_ x, (h0, c0))
        # 对LSTM输出进行解包
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs,
                                                padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        # 融合双向LSTM的维度
        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1)\
                        .transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask \
                if encoder_padding_mask.any() else None
        }

# 注意力机制
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim,
                 output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim,
                                 source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim,
                                 output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):

        # 假设input为 B×H，B为迷你批次的大小，H为隐含状态大小
        # 假设source_hids为L×B×H，L为序列长度，B为迷你批次的大小，
        # H为隐含状态大小
        x = self.input_proj(input)

        # 计算注意力分数
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # 设置填充单词的注意力分数为-inf
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)

        attn_scores = F.softmax(attn_scores, dim=0)

        # 对编码器输出加权平均
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores

# 解码器
class LSTMDecoder(nn.Module):

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512):

        super(LSTMDecoder, self).__init__()
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.need_attn = True

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units,
                                              hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim \
                    if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        self.attention = AttentionLayer(hidden_size, encoder_output_units,
                                        hidden_size, bias=False)
        self.fc_out = Linear(out_embed_dim, num_embeddings,
                             dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, 
                incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        # 获取保存的输出单词，用于模型的预测
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # 获取保存的状态，用于模型的预测
        cached_state = utils.get_incremental_state(self, incremental_state,
                                                      'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) \
                                for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []

        # 进行迭代的循环神经网络计算
        for j in range(seqlen):
            # 输入中引入上一步的注意力机制的信息
            input = torch.cat((x[j, :, :], input_feed), dim=1)
            # 迭代所有的循环神经网络层
            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                input = F.dropout(hidden, p=self.dropout_out,
                                  training=self.training)

                prev_hiddens[i] = hidden
                prev_cells[i] = cell
            # 计算注意力的输出值和注意力权重
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)
            input_feed = out
            outs.append(out)
            
        # 保存隐含状态
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        attn_scores = attn_scores.transpose(0, 2)
        x = self.fc_out(x)
        return x, attn_scores
