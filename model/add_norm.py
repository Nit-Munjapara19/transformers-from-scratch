import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer_output):
        sublayer_output = self.dropout(sublayer_output)
        x = x + sublayer_output
        x = self.layer_norm(x)
        return x