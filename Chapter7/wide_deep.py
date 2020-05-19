""" 本代码仅作为Wide&Deep模型的实现参考
"""

import torch
import torch.nn as nn

class WideDeep(nn.Module):
    def __init__(self, num_wide_feat, deep_feat_sizes, 
        deep_feat_dims, nhiddens):

        super(WideDeep, self).__init__()

        self.num_wide_feat = num_wide_feat
        self.deep_feat_sizes = deep_feat_sizes
        self.deep_feat_dims = deep_feat_dims
        self.nhiddens = nhiddens

        # 深模型的嵌入部分
        self.embeds = nn.ModuleList()
        for deep_feat_size, deep_feat_dim in \
            zip(deep_feat_sizes, deep_feat_dims):
            self.embeds.append(nn.Embedding(deep_feat_size, 
                deep_feat_dim))

        self.deep_input_size = sum(deep_feat_dims)

        # 深模型的线性部分 
        self.linears = nn.ModuleList()
        in_size = self.deep_input_size
        for out_size in nhiddens:
            self.linears.append(nn.Linear(in_size, out_size))
            in_size = out_size

        # 宽模型和深模型共同的线性部分 
        self.proj = nn.Linear(in_size + num_wide_feat, 1)

    def forward(self, wide_input, deep_input):
        
        # 假设宽模型的输入为N×W，N为迷你批次的大小，W为宽特征的大小
        # 假设深模型的输入为N×D，N为迷你批次的大小，D为深特征的数目
        embed_feats = []
        for i in range(deep_input.size(1)):
            embed_feats.append(self.embeds[i](deep_input[:, i]))
        deep_feats = torch.cat(embed_feats, 1)
        
        # 深模型特征变换
        for layer in self.linears:
            deep_feats = layer(deep_feats)
            deep_feats = torch.relu(deep_feats)
        print(wide_input.shape, deep_feats.shape)

        # 宽模型和深模型特征拼接
        wide_deep_feats = torch.cat([wide_input, deep_feats], -1)
        return torch.sigmoid(self.proj(wide_deep_feats)).squeeze()
