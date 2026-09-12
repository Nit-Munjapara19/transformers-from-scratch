import torch
import torch.nn as nn
from model.decoder_layer import DecoderLayer
from model.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self,vocabulary_size,d_model,num_heads,num_layers,d_ff,dropout,max_sequence_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model,max_sequence_length=max_sequence_length,dropout=dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self,token_ids,encoder_output,self_attention_mask=None,cross_attention_mask=None):
        x = self.embedding(token_ids)
        x = x * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        all_self_attention_weights = []
        all_cross_attention_weights = []

        for layer in self.layers:
            (x,self_attention_weights,cross_attention_weights) = layer(x=x,encoder_output=encoder_output,self_attention_mask=self_attention_mask,cross_attention_mask=cross_attention_mask)
            all_self_attention_weights.append(self_attention_weights)
            all_cross_attention_weights.append(cross_attention_weights)
        x = self.final_layer_norm(x)

        return (x,all_self_attention_weights,all_cross_attention_weights)