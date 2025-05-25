import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .attention import FFN

class EncoderLayer(nn.Module):
    def __init__(self, 
                 h:int,
                 d:int,
                 d_k:int,
                 d_ff:int,
                 p_dropout:float=0.1):
        
        super().__init__()

        # attributes
        self.h = h
        self.d = d
        self.d_k = d_k
        self.dropout = nn.Dropout(p_dropout)

        # multi-head self-attention
        self.mhsa = MultiHeadAttention(h=h, d=d, d_k=d_k, d_v=d_k)

        # FFN
        self.ffn = FFN(d=d, d_ff=d_ff)

        # layer normalization
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)

    def forward(self, 
                x:torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        """

        # MHSA
        x = self.norm1(x + self.dropout(self.mhsa(x, mask=mask)))

        # FNN
        x = self.norm2(x + self.dropout(self.ffn(x)))
        
        return x