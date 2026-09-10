import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query,key,value,mask=None):
    key_dimension = query.size(-1)
    attention_scores = torch.matmul(query,key.transpose(-2,-1))
    attention_scores = attention_scores/math.sqrt(key_dimension)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0,float('-inf'))

    attention_weights = F.softmax(attention_scores,dim=-1)
    attention_output = torch.matmul(attention_weights,value)

    return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()

        if d_model%num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads

        self.query_projection = nn.Linear(d_model,d_model)
        self.key_projection = nn.Linear(d_model,d_model)
        self.value_projection = nn.Linear(d_model,d_model)
        self.output_projection = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

        def forward(self,query,key,value,mask=None):
            batch_size = query.size(0)
            query = self.query_projection(query)
            key = self.key_projection(key)
            value = self.value_projection(value)

            query = query.view(batch_size,-1,self.num_heads,self.head_dimension).transpose(1, 2)
            #view --> changes: [batch,sequence,256] into [batch,sequence,8,32]
            key = key.view(batch_size,-1,self.num_heads,self.head_dimension).transpose(1, 2)
            value = value.view(batch_size,-1,self.num_heads,self.head_dimension).transpose(1, 2)
    
            attention_output, attention_weights = (scaled_dot_product_attention(query,key,value,mask))
            attention_output = attention_output.transpose(1,2)
            attention_output = attention_output.contiguous().view(batch_size,-1,self.d_model)
            attention_output = self.output_projection(attention_output)
            attention_output = self.dropout(attention_output)
    
            return attention_output, attention_weights