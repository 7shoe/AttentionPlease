# decoder.py
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
                 p_dropout:float=0.0,
                 pre_layer_norm:bool=False):
        
        super().__init__()

        # attributes
        self.h = h
        self.d = d
        self.d_k = d_k
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(self.p_dropout)
        self.pre_layer_norm = pre_layer_norm

        # Self- & cross-Attention
        self.self_attn  = MultiHeadAttention(h=self.h, d=self.d, d_k=self.d_k, d_v=self.d_k, p_dropout=self.p_dropout)
        self.cross_attn = MultiHeadAttention(h=self.h, d=self.d, d_k=self.d_k, d_v=self.d_k, p_dropout=self.p_dropout)

        # FFN
        self.ffn = FFN(d=self.d, d_ff=self.d_ff)

        # layer normalization
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
        self.norm3 = nn.LayerNorm(self.d)
        # pre-LN: final normalization
        if self.pre_layer_norm:
            self.norm4 = nn.LayerNorm(self.d)

    def forward(self, 
                X_tgt:torch.Tensor, 
                X_enc:torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        """

        # 1) MH Self-Attention
        if self.pre_layer_norm:
            X_tgt = self.norm1(X_tgt)
            X_tgt = X_tgt + self.dropout(self.self_attn(Q_in=X_tgt, mask=self_attn_mask))
        else:
            X_tgt = self.norm1(X_tgt + self.dropout(self.self_attn(Q_in=X_tgt, mask=self_attn_mask)))

        # 2) MH Cross-Attention
        if self.pre_layer_norm:
            X_tgt = self.norm2(X_tgt)
            X_tgt = X_tgt + self.dropout(self.cross_attn(Q_in=X_tgt, K_in=X_enc, V_in=X_enc, mask=cross_attn_mask))
        else:
            X_tgt = self.norm2(X_tgt + self.dropout(self.cross_attn(Q_in=X_tgt, K_in=X_enc, V_in=X_enc, mask=cross_attn_mask)))

        # 3) FFN
        if self.pre_layer_norm:
            X_tgt = self.norm3(X_tgt)
            X_tgt = X_tgt + self.dropout(self.ffn(X_tgt))
        else:
            X_tgt = self.norm3(X_tgt + self.dropout(self.ffn(X_tgt)))

        # 4) Otional: final normalization
        if self.pre_layer_norm:
            X_tgt = self.norm4(X_tgt)
        
        return X_tgt