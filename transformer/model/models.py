import torch
import torch.nn as nn
from typing import Optional
import math

from .encoder import EncoderLayer
from .decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self,
                 N:int,
                 L:int,
                 h:int,
                 d:int,
                 d_ff:int,
                 n_vocab:int,
                 padding_idx:int,
                 bos_idx:int,
                 dtype:torch.dtype,
                 device:torch.device):

        super().__init__()

        self.N = N
        self.L = L # in TraFo
        self.d = d # in Trafo
        self.h = h
        self.d_k = self.d // h
        self.d_ff = d_ff
        self.n_vocab = n_vocab
        self.bos_idx = bos_idx
        self.padding_idx = padding_idx
        self.dtype = dtype
        self.device = device

        # shared embedding
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab,
                                     embedding_dim=self.d,
                                     padding_idx=self.padding_idx)

        # Encoder
        # - token id embedding
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d=self.d, h=self.h, d_k=self.d_k, d_ff=self.d_ff) 
            for _ in range(self.N)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(h=self.h, d=self.d, d_k=self.d_k, d_ff=self.d_ff)
            for _ in range(self.N)
        ])

        # Output projection
        self.output_head = nn.Linear(self.d, self.n_vocab, bias=False)

        # -> device
        self.to(self.device)

        pass

    def forward(self, 
                src_ids:  torch.LongTensor,
                tgt_ids:  torch.LongTensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None):
        """
        Forward
        """

        # device
        src_ids = src_ids.to(self.device)
        tgt_ids = tgt_ids.to(self.device)

        # embedd
        X_enc = self.embedding(src_ids)
        X_enc = self.positional_encoding(X_enc)

        # encode
        for encoder_layer in self.encoder_layers:
            X_enc = encoder_layer(X_enc, src_mask)

        # decode
        X_dec = self.embedding(self.right_shift(tgt_ids))
        X_dec = self.positional_encoding(X_dec)

        # causal mask for decoder
        if tgt_mask is None:
            tgt_mask = self.causal_mask()
        
        for decoder_layer in self.decoder_layers:
            X_dec = decoder_layer(X_dec, X_enc, tgt_mask)

        # proj into vocab space
        logits = self.output_head(X_dec)

        return logits

    def causal_mask(self,):
        """
        Upper triangular
        """
        
        mask = torch.triu(torch.ones(self.L, self.L, device=self.device), diagonal=1).bool()
        
        return mask

    def right_shift(self, X: torch.Tensor):
        """
        Right-shift the last dimension of X, padding in bos_idx at the front.
        Works for X of shape (B, L) or (L,).
        """
        # roll token ids 1 step to the right
        Xr = torch.roll(X, shifts=1, dims=-1)
        
        # assign <BOS> token ID
        Xr[..., 0] = self.bos_idx
        
        return Xr


    def positional_encoding(self, X:torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding:
        
        - $PE(pos,2i)   = sin(pos / 10_000^{2i / d})$
        - $PE(pos,2i+1) = cos(pos / 10_000^{2i / d})$
        
        for pos as token sequence index in {0,...,L-1} and i as embedding index in {0,...,d-1}.
        """

        L, d = X.size()[-2], X.size()[-1]
        
        # broadcast
        pos = torch.arange(L, device=self.device, dtype=self.dtype).unsqueeze(1)
        i = torch.arange(d, device=self.device, dtype=self.dtype).unsqueeze(0)
    
        # divisor
        divisor = (10_000 ** ((2.0 * i) / self.d))
        
        # PosEnc (even: sin, odd: cos)
        PE = torch.zeros(L, d, device=self.device, dtype=self.dtype)
        PE[:, 0::2] = torch.sin(pos / divisor[:, 0::2])
        PE[:, 1::2] = torch.cos(pos / divisor[:, 1::2])

        return X + PE