import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .attention import FFN

class DecoderLayer(nn.Module):
    def __init__(self, 
                 h:int,
                 d:int,
                 d_k:int,
                 d_ff:int,
                 p_dropout:float=0.01):
        
        super().__init__()

        # attributes
        self.h = h
        self.d = d
        self.d_k = d_k
        self.dropout = nn.Dropout(p_dropout)

        # Self- & cross-Attention
        self.self_attn  = MultiHeadAttention(h=h, d=d, d_k=d_k, d_v=d_k)
        self.cross_attn = MultiHeadAttention(h=h, d=d, d_k=d_k, d_v=d_k)

        # FFN
        self.ffn = FFN(d=d, d_ff=d_ff)

        # layer normalization
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, 
                X_tgt:torch.Tensor, 
                X_enc:torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        """

        # 1)
        X_tgt = self.norm1(X_tgt + self.dropout(self.self_attn(X_tgt, mask=mask)))

        # 2)
        X_tgt = self.norm2(X_tgt + self.dropout(self.cross_attn(X_tgt, K_in=X_enc, V_in=X_enc)))

        # 3) FFN
        X_tgt = self.norm3(X_tgt + self.dropout(self.ffn(X_tgt)))
        
        return X_tgt