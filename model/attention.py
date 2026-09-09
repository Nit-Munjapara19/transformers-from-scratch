import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query,key,value,mask=None):
    key_dimension = query.size(-1)
    attention_scores = torch.matmul(query,key.transpose(-2,-1))
    attention_scores = attention_scores/math.sqrt(key_dimension)

    attention_weights = F.softmax(attention_scores,dim=-1)
    attention_output = torch.matmul(attention_weights,value)

    return attention_output, attention_weights