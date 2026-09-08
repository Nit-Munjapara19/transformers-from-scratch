import torch
import torch.nn as nn

# What is nn.Module ? nn.Module is base class that gives your model useful PyTorch functionality.
# For example, it allows PyTorch to:
# keep track of trainable parameters
# move the model to GPU
# switch between training/evaluation modes
# save/load model parameters
# work with optimizers
# compose multiple layers together

class TokenEmbedding(nn.Module):
    def __init__(self,vocabulary_size,d_model): # embedding vector has D_MODEL values
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,embedding_dim=d_model)

    def forward(self,token_ids):
        return self.embedding(token_ids)