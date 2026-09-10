import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()

        self.linear_1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x