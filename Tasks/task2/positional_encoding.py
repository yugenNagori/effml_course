import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        
        PE = torch.zeros((max_len, d_model))
        pairs = torch.arange(0, d_model, 2) # номер пары
        div = 10000 ** (pairs / d_model) # 2*pairs если бы шли до d_model // 2
        pos = torch.arange(max_len).reshape(-1, 1)
        PE[:, ::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("PE", PE.unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        t, b, e = x.shape
        PE = self.PE[:t]
        return self.dropout(x + PE)
        

        
