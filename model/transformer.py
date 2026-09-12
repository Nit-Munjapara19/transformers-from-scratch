import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.output_projection import OutputProjection

class Transformer(nn.Module):
    def __init__(self,source_vocabulary_size,target_vocabulary_size,d_model,num_heads,num_encoder_layers,num_decoder_layers,d_ff,dropout,max_sequence_length):
        super().__init__()

        self.encoder = Encoder(vocabulary_size=source_vocabulary_size,d_model=d_model,num_heads=num_heads,num_layers=num_encoder_layers,d_ff=d_ff,dropout=dropout,max_sequence_length=max_sequence_length)

        self.decoder = Decoder(vocabulary_size=target_vocabulary_size,d_model=d_model,num_heads=num_heads,num_layers=num_decoder_layers,d_ff=d_ff,dropout=dropout,max_sequence_length=max_sequence_length)

        self.output_projection = OutputProjection(d_model=d_model,vocabulary_size=target_vocabulary_size)

    def forward(self,source_token_ids,target_token_ids,source_mask=None,target_self_attention_mask=None,target_cross_attention_mask=None):
        encoder_output, encoder_attention_weights = self.encoder(token_ids=source_token_ids,mask=source_mask)

        (decoder_output,decoder_self_attention_weights,decoder_cross_attention_weights) = self.decoder(token_ids=target_token_ids,encoder_output=encoder_output,self_attention_mask=target_self_attention_mask,cross_attention_mask=target_cross_attention_mask)

        logits = self.output_projection(decoder_output)
        return (logits,encoder_attention_weights,decoder_self_attention_weights,decoder_cross_attention_weights)