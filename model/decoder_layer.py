import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForwardNetwork
from model.add_norm import AddNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads,dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads,dropout=dropout)

        self.feed_forward = FeedForwardNetwork(d_model=d_model,d_ff=d_ff,dropout=dropout)
        self.add_norm_1 = AddNorm(d_model=d_model,dropout=dropout)
        self.add_norm_2 = AddNorm(d_model=d_model,dropout=dropout)
        self.add_norm_3 = AddNorm(d_model=d_model,dropout=dropout)

    def forward(self,x,encoder_output,self_attention_mask=None,cross_attention_mask=None):
        self_attention_output, self_attention_weights = (
            self.self_attention(query=x,key=x,value=x,mask=self_attention_mask)
        )

        x = self.add_norm_1(x,self_attention_output)

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(query=x,key=encoder_output,value=encoder_output,mask=cross_attention_mask)
        )
        x = self.add_norm_2(x,cross_attention_output)

        feed_forward_output = self.feed_forward(x)
        x = self.add_norm_3(x,feed_forward_output)
        return (x,self_attention_weights,cross_attention_weights)