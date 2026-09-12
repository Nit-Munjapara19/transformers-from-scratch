import torch
import torch.nn as nn

class OutputProjection(nn.Module):
    def __init__(self, d_model, vocabulary_size):
        super().__init__()
        self.linear = nn.Linear(d_model,vocabulary_size)

    def forward(self, decoder_output):
        logits = self.linear(decoder_output)
        return logits