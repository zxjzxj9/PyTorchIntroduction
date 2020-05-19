""" 以下代码仅作为word2vec的CBOW模型的实现参考
"""

import torch
import torch.nn as nn

# 定义CBOW模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(context_size*embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs):
        embedded = self.embeddings(inputs)
        embedded = embedded.view(embedded.size(0), -1)
        hid = torch.relu(self.linear1(embedded))
        out = self.linear2(hid)
        return out


# 模型的训练
def train_cbow():
    hidden_size = 128
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    model = CBOW(vocab_size, embd_size, context_size, hidden_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        for context, target in cbow_train:
            model.zero_grad()
            logits = model(context)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

    return model

