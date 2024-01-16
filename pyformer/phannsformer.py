import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[0,:x.size(1),:]

class Phannsformer(nn.Module):
    def __init__(self, embedding_size=64, heads=8, layers=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, heads, embedding_size*2, dropout=0.1, activation='relu', batch_first=True)
        self.base = nn.Sequential(
            nn.Embedding(25, embedding_size),
            PositionalEncoding(embedding_size),
            nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        )
        self.denoiser = nn.Linear(embedding_size, 25)
        self.classifier = nn.Linear(embedding_size, 12)

    def classify(self, x):
        base = self.base(x)
        return self.classifier(base)

    def denoise(self, x):
        base = self.base(x)
        return self.denoiser(base)
