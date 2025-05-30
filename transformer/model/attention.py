# attention.py
import math
import torch
import torch.nn as nn
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 h:int,
                 d:int, 
                 d_k:int, 
                 d_v:int,
                 p_dropout:float=0.0):
        
        super().__init__()
        
        self.h = h
        self.d = d
        self.d_k = d_k
        self.d_v = d_v
        self.p_dropout = p_dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(in_features=self.d,        out_features=self.h * self.d_k, bias=False)
        self.k_proj = nn.Linear(in_features=self.d,        out_features=self.h * self.d_k, bias=False)
        self.v_proj = nn.Linear(in_features=self.d,        out_features=self.h * self.d_v, bias=False)
        self.out_proj= nn.Linear(in_features=self.h * self.d_v, out_features=self.d,       bias=False)

        # attention dropout
        self.dropout = nn.Dropout(self.p_dropout)
        
        pass

    def split_heads(self, tensor: torch.Tensor, d_head: int):
        """
        Split tensor into multiple heads
        """
        *lead, L, _ = tensor.shape
        reshaped = tensor.reshape(*lead, L, self.h, d_head)
            
        # Move head dimension: (*lead, L, h, d_head) -> (*lead, h, L, d_head)
        perm = list(range(len(lead))) + [len(lead)+1, len(lead), len(lead)+2]
        
        return reshaped.permute(*perm)

    def combine_heads(self, tensor: torch.Tensor):
        """
        Combine multiple heads back into single representation
        Input:  (*lead, h, L, d_head)
        Output: (*lead, L, h*d_head)
        """
        *lead, h, L, d_head = tensor.shape
        
        # Move head dimension: (*lead, h, L, d_head) -> (*lead, L, h, d_head)
        perm = list(range(len(lead))) + [len(lead)+1, len(lead), len(lead)+2]
        permuted = tensor.permute(*perm)
        
        return permuted.reshape(*lead, L, h * d_head)


    def forward(self,
                Q_in: torch.Tensor,
                K_in: Optional[torch.Tensor]=None,
                V_in: Optional[torch.Tensor]=None,
                mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        MHA forward pass
        """

        # Default to self-attention if K, V not provided
        if K_in is None:
            K_in = Q_in
        if V_in is None:
            V_in = Q_in
            
        # Q,K,V projection
        Q = self.q_proj(Q_in)
        K = self.k_proj(K_in)
        V = self.v_proj(V_in)

        # split out heads
        Q = self.split_heads(Q, self.d_k)     # (B, h, L, d_k)
        K = self.split_heads(K, self.d_k)     # (B, h, L, d_k)
        V = self.split_heads(V, self.d_v)     # (B, h, L, d_v)
        
        # Q @ K.T
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k) # (-1,-2)
        
        # mask
        if mask is not None:
            # Handle different mask shapes robustly
            if mask.dim() == 2:  # (B, L) -> for key masking
                # Expand to (B, 1, 1, L) for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # (B, L_q, L_k) -> full attention mask
                # Expand to (B, 1, L_q, L_k) for broadcasting
                mask = mask.unsqueeze(1)

            scores = scores.masked_fill(mask, -1e9)

        # softmax()
        attn = torch.softmax(scores, dim=-1)
        #attn = self.dropout(attn)  # Add this
        context = torch.matmul(attn, V)

        # recombine heads
        context = self.combine_heads(context)  # (*lead, L, h*d_v)

        # output projecion
        output = self.out_proj(context)

        return output

class FFN(nn.Module):
    def __init__(self, d:int, d_ff:int):
        super().__init__()
        self.d=d
        self.d_ff = d_ff

        self.layer_1 = nn.Linear(in_features=self.d, out_features=self.d_ff)
        self.layer_2 = nn.Linear(in_features=self.d_ff, out_features=self.d)
        self.nonlinear = nn.ReLU()

    def forward(self, x:torch.Tensor):

        return self.layer_2(self.nonlinear(self.layer_1(x)))