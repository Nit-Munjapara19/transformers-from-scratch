import torch

def create_padding_mask(token_ids, pad_token_id):
    mask = (token_ids != pad_token_id)
    return mask

def create_causal_mask(sequence_length):
    mask = torch.tril(torch.ones(sequence_length,sequence_length,dtype=torch.bool))
    return mask