import torch

def create_padding_mask(token_ids, pad_token_id):
    return token_ids != pad_token_id

def create_causal_mask(sequence_length, device=None):
    return torch.tril(torch.ones(sequence_length,sequence_length,dtype=torch.bool,device=device))

def create_decoder_self_attention_mask(token_ids,pad_token_id):
    batch_size, sequence_length = token_ids.shape
    padding_mask = create_padding_mask(token_ids,pad_token_id)
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    causal_mask = create_causal_mask(sequence_length,device=token_ids.device)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    mask = padding_mask & causal_mask
    return mask