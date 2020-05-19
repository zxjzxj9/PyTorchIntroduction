""" 以下代码仅作为情感分析模型的实现参考
"""

import torch
import torch.nn as nn

class Sentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        
        super(Sentiment, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), im = 1))     
        return self.fc(hidden)


model = Sentiment(vocab_size, embedding_dim, hidden_dim, output_dim,
                     n_layers, bidirectional, dropout, pad_idx)
optimizer = optim.Adam(model.parameters(), lr=lr)
def train(model, iterator, optimizer, criterion):
    
    model.train()
    
    for batch in iterator:        
        optimizer.zero_grad()
        text, text_lengths = batch.text  
        predictions = model(text, text_lengths).squeeze(1)   
        loss = criterion(predictions, batch.label)    
        loss.backward()       
        optimizer.step()
        
    return model

