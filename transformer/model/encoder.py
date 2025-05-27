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
                 p_dropout:float=0.0,
                 pre_layer_norm:bool=False):
        """
        - pre_layer_norm follows suggestion of Pre-LN Transformer (https://arxiv.org/pdf/2002.04745)
        """
        
        super().__init__()

        # attributes
        self.h = h
        self.d = d
        self.d_k = d_k
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(self.p_dropout)
        self.pre_layer_norm = pre_layer_norm

        # multi-head self-attention
        self.mhsa = MultiHeadAttention(h=self.h, d=self.d, d_k=self.d_k, d_v=self.d_k, p_dropout=self.p_dropout)

        # FFN
        self.ffn = FFN(d=d, d_ff=d_ff)

        # layer normalization
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
        # pre-LN: final normalization
        if self.pre_layer_norm:
            self.norm3 = nn.LayerNorm(self.d)

    def forward(self, 
                x:torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        """

        # MHSA
        if self.pre_layer_norm:
            x = self.norm1(x)
            x = x + self.dropout(self.mhsa(x, mask=mask))
        else:
            x = self.norm1(x + self.dropout(self.mhsa(x, mask=mask)))

        # FNN
        if self.pre_layer_norm:
            x = self.norm2(x)
            x = x + self.dropout(self.ffn(x))
        else:
            x = self.norm2(x + self.dropout(self.ffn(x)))

        # optional: final normalization
        if self.pre_layer_norm:
            x = self.norm3(x)
        
        return x