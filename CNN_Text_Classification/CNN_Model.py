# encoding='utf-8'

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class CNNNet(nn.Module):
    # code here

    def __init__(self, vocab_size=-1, word_embedding_size=300, filter_sizes=(3, 4, 5), batch_size=64, dropout_keep_prob=0.5):
        # code here
        super(CNNNet, self).__init__()
        self.filter_sizes = filter_sizes
        self.vocab_size = vocab_size
        self.word_embedding_size = word_embedding_size
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob

        self.convs = nn.ModuleList([nn.Conv1d(word_embedding_size, 1, filter_size) for filter_size in filter_sizes])
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.fc1 = nn.Linear(len(filter_sizes), 5)

    def forward(self, x):
        x = [F.relu(conv(x)) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

