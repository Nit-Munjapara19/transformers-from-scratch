import math
import torch
import torch.nn as nn

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_into_heads(self, x):
        batch_size, sequence_length, _ = x.shape
        x = x.view(batch_size, sequence_length, self.num_heads, self.head_dimension)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(2)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, sequence_length, self.d_model)

    def forward(self, query, key, value, mask=None):
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        query = self.split_into_heads(query)
        key = self.split_into_heads(key)
        value = self.split_into_heads(value)
        attention_output, attention_weights = scaled_dot_product_attention(query=query, key=key, value=value, mask=mask)
        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)
        return attention_output, attention_weights