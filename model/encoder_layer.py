import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForwardNetwork
from model.add_norm import AddNorm

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads,dropout=dropout)
        self.feed_forward =  FeedForwardNetwork(d_model=d_model,d_ff=d_ff,dropout=dropout)
        self.add_norm_1 = AddNorm(d_model=d_model,dropout=dropout)
        self.add_norm_2 = AddNorm(d_model=d_model,dropout=dropout)

    def forward(self,x,mask=None):
        attention_output, attention_weights = self.self_attention(query=x,key=x,value=x,mask=mask)
        x = self.add_norm_1(x,attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.add_norm_2(x,feed_forward_output)
        return x, attention_weights