import torch
import torch.nn as nn
from model.encoder_layer import EncoderLayer
from model.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self,vocabulary_size,d_model,num_heads,num_layers,d_ff,dropout,max_sequence_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model,max_sequence_length=max_sequence_length,dropout=dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
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

    def forward(self, token_ids, mask=None):
        x = self.embedding(token_ids)
        x = x * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x,mask=mask)
            all_attention_weights.append(attention_weights)
        x = self.final_layer_norm(x)
        return x, all_attention_weights