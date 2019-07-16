#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
__all__ = ['Med2Vec']

class Med2Vec(BaseModel):
    def __init__(self, icd9_size, demographics_size=0, embedding_size=2000, hidden_size=100,):
        super(Med2Vec, self).__init__()
        self.embedding_size = embedding_size
        self.demographics_size = demographics_size
        self.hidden_size = hidden_size
        self.vocabulary_size = icd9_size
        self.embedding_demo_size = self.embedding_size + self.demographics_size
        self.embedding_w = torch.nn.Parameter(torch.Tensor(self.embedding_size, self.vocabulary_size))
        torch.nn.init.uniform_(self.embedding_w, a=-0.1, b=0.1)
        self.embedding_b = torch.nn.Parameter(torch.Tensor(1, self.embedding_size))
        self.embedding_b.data.fill_(0)
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(self.embedding_demo_size, self.hidden_size)
        self.probits = nn.Linear(self.hidden_size, self.vocabulary_size)

        self.bce_loss = nn.BCEWithLogitsLoss()


    def embedding(self, x):
        return F.linear(x, self.embedding_w, self.embedding_b)

    def forward(self, x, d=torch.Tensor([])):
        x = self.embedding(x)
        x = self.relu1(x)
        emb = F.relu(self.embedding_w)

        if (self.demographics_size):
            x = torch.cat((x, d), dim=1)
        x = self.linear(x)
        x = self.relu2(x)
        probits = self.probits(x)
        return probits, emb
