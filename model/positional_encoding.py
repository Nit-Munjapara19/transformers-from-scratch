import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_sequence_length,dropout):
        self.dropout = nn.Dropout(dropout) #Dropout randomly turns off some values in a neural network during training.The main reason is to prevent overfitting
        positional_encoding = torch.zeros(max_sequence_length,d_model)

        position = torch.arange(0,max_sequence_length-1,
                                dtype=float).unsqueeze(1) #unsqueeze change dimensional [max_sequence_length] --> [max_sequence_length,1]

        div_term = torch.exp(
            torch.arange(0,d_model,2,dtype=float)*(-math.log(10000.0)/d_model)
        )

        positional_encoding[:,0::2] = torch.sin(position*div_term)
        positional_encoding[:,1::2] = torch.cos(position*div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        # register_buffer means:
        # - PyTorch keeps this tensor with the model
        # - it moves to GPU with the model
        # - it is saved with the model
        # - but it is NOT a trainable parameter
        self.register_buffer("positional_encoding",positional_encoding)

        def forward(self,embeddings):
            sequence_length = embeddings.size(1)
            embeddings = embeddings + self.positional_encoding[:,sequence_length,:]
            return self.dropout(embeddings)