# models.py
import torch
import torch.nn as nn
from typing import Optional, Union
from pathlib import Path
import math
import yaml

from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .generator import GeneratorLayer

class Transformer(nn.Module):
    def __init__(self,
                 N:int = None,
                 L_max:int = None,
                 h:int = None,
                 d:int = None,
                 d_ff:int = None,
                 p_dropout:float=None,
                 n_vocab:int = None,
                 padding_idx:int = None,
                 bos_idx:int = None,
                 dtype:torch.dtype = None,
                 device:torch.device = None,
                 pre_trained: bool = False,
                 yaml_path: Union[str, Path] = None):
        """
        Args:
            pre_trained: If True, load model from yaml_path
            yaml_path: Path to YAML file containing model config and checkpoint path
            
        Usage:
            # Create new model
            model = Transformer(N=6, L_max=64, h=8, d=512, ...)
            
            # Load pre-trained model
            model = Transformer(pre_trained=True, yaml_path="./logs/1748295195.yaml")
        """

        super().__init__()

        # pre-trained
        if pre_trained:
            if yaml_path is None:
                raise ValueError("yaml_path must be provided when pre_trained=True")
            
            # Load YAML file
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Extract model config from args
            args_dict = yaml_data['args']
            
            # Set model attributes
            self.N = args_dict['N']
            self.L_max = args_dict['L_max']
            self.d = args_dict['d']
            self.h = args_dict['h']
            self.d_k = self.d // self.h
            self.d_ff = args_dict['d_ff']
            self.p_dropout = args_dict['p_dropout']
            self.n_vocab = args_dict['n_vocab']
            self.bos_idx = args_dict['bos_idx']
            self.padding_idx = args_dict['padding_idx']
            
            # Set system parameters
            self.dtype = dtype or torch.float32
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            # Build model architecture first
            self._build_model_architecture()
            
            # Load checkpoint weights
            if 'metadata' in yaml_data and 'checkpoint' in yaml_data['metadata']:
                checkpoint_path = yaml_data['metadata']['checkpoint']
                self._load_checkpoint_weights(checkpoint_path)
                print(f"✅ Model loaded from: {yaml_path}")
                print(f"✅ Weights loaded from: {checkpoint_path}")
            else:
                print(f"⚠️  No checkpoint path found in YAML - using random weights")
        
        else:
            # Original new model creation
            self.N = N
            self.L_max = L_max
            self.d = d
            self.h = h
            self.d_k = self.d // h
            self.d_ff = d_ff
            self.p_dropout = p_dropout
            self.n_vocab = n_vocab
            self.bos_idx = bos_idx
            self.padding_idx = padding_idx
            self.dtype = dtype
            self.device = device
            
        # build model
        self._build_model_architecture()

        # generator
        self._generator = None

        # DEBUG
        self.encoder_src_mask = None
        self.tgt_pad_mask = None
        self.X_before = None
        self.X_after = None

        
        pass
        

    @property
    def generator(self):
        """Lazy initialization of generator"""
        if self._generator is None:
            self._generator = GeneratorLayer(self, max_length=self.L_max)
        return self._generator
    
    @torch.no_grad()
    def generate(self, 
                 src_ids: torch.LongTensor,
                 L_max: int = 64,
                 eos_idx: int = 2,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 do_sample: bool = True) -> torch.LongTensor:
        """
        Generate text autoregressively
        
        Args:
            src_ids: Source token IDs [batch_size, src_len]
            L_max: Maximum target sequence length
            eos_idx: End-of-sequence token ID
            temperature: Sampling temperature (1.0 = neutral, <1.0 = focused, >1.0 = diverse)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold (e.g., 0.9)
            do_sample: Use sampling if True, greedy decoding if False
            
        Returns:
            Generated token IDs [batch_size, generated_len]
            
        Example:
            generated_ids = model.generate(
                src_ids=batch['src_ids'],
                L_max=64,
                eos_idx=tokenizer.token_vocab['<EOS>'],
                temperature=0.8,
                top_p=0.9
            )
        """
        return self.generator.generate(
            src_ids=src_ids,
            L_max=L_max,
            eos_idx=eos_idx,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
    
    def beam_search(self, 
                   src_ids: torch.LongTensor,
                   L_max: int = 64,
                   eos_idx: int = 2,
                   beam_size: int = 4):
        """
        Beam search decoding for better quality
        
        Returns:
            List of generated sequences, sorted by score
        """
        return self.generator.beam_search(
            src_ids=src_ids,
            L_max=L_max,
            eos_idx=eos_idx,
            beam_size=beam_size
        )

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

        # infer size
        B, L_src = src_ids.size()
        _, L_tgt = tgt_ids.size()
        
        # embedd (scaling + pos. encoding)
        X_enc = self.embedding(src_ids) 
        X_enc = X_enc * math.sqrt(self.d) + self.PE[:L_src]

        # source mask (for encoder)
        if src_mask is None:
            src_pad_mask = src_ids.eq(self.padding_idx)        # (B, L_src)
            src_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_src)

        # DEBUG
        #self.encoder_src_mask = src_mask # confirmed correct

        # encode
        for encoder_layer in self.encoder_layers:
            X_enc = encoder_layer(X_enc, 
                                  mask=src_mask)

        # target mask (for decoder self-attention only)
        if tgt_mask is None:
            # Get the right-shifted sequence for proper mask alignment
            shifted_tgt = self.right_shift(tgt_ids)
            
            # padding mask based on right-shifted sequence
            tgt_pad_mask = shifted_tgt.eq(self.padding_idx)  # (B, L_tgt)

            # causal mask
            causal = self.causal_mask(L_tgt)  # (L_tgt, L_tgt)
            causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, L_tgt, L_tgt)
            
            # Target key mask (for self-attention)
            tgt_key_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_tgt)
            
            # Combine for self-attention
            tgt_mask = causal | tgt_key_mask  # (B, 1, L_tgt, L_tgt)

            self.tgt_mask = tgt_mask.detach().cpu()
        
        # cross attention mask (for decoder cross-attention)
        cross_attn_mask = src_ids.eq(self.padding_idx).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L_src)

        # decode (embedding scaling & pos. encoding)
        X_dec = self.embedding(self.right_shift(tgt_ids)) * math.sqrt(self.d) + self.PE[:L_tgt]
        
        for decoder_layer in self.decoder_layers:
            X_dec = decoder_layer(X_dec, 
                                  X_enc, 
                                  self_attn_mask=tgt_mask,
                                  cross_attn_mask=cross_attn_mask)

        # proj into vocab space
        logits = self.output_head(X_dec)

        return logits

    def causal_mask(self, L_tgt:int):
        """
        Upper triangular
        """
        
        mask = torch.triu(torch.ones(L_tgt, L_tgt, device=self.device), diagonal=1).bool()

        # DEBUG
        self.causal_m = mask.detach().cpu()
        
        return mask

    def right_shift(self, X: torch.Tensor):
        """
        Right-shift the last dimension of X, padding in bos_idx at the front.
        Works for X of shape (B, L) or (L,).
        """

        # DEBUG
        self.X_before = X.detach().cpu()
        
        # roll token ids 1 step to the right
        Xr = torch.roll(X, shifts=1, dims=-1)
        
        # assign <BOS> token ID
        Xr[..., 0] = self.bos_idx

        # DEBUG
        self.X_after = Xr.detach().cpu()
        
        return Xr


    def positional_encoding(self) -> None:
        """
        Add positional encoding:
        
        - $PE(pos,2i)   = sin(pos / 10_000^{2i / d})$
        - $PE(pos,2i+1) = cos(pos / 10_000^{2i / d})$
        
        for pos as token sequence index in {0,...,L-1} and i as embedding index in {0,...,d-1}.
        """
        
        # pos, i
        pos = torch.arange(self.L_max, device=self.device, dtype=self.dtype).unsqueeze(1)
        i = torch.arange(self.d, device=self.device, dtype=self.dtype).unsqueeze(0)
    
        # divisor
        divisor = (10_000 ** ((2.0 * i) / self.d))
        
        # PosEnc (even: sin, odd: cos)
        PE = torch.zeros(self.L_max, self.d, device=self.device, dtype=self.dtype)
        PE[:, 0::2] = torch.sin(pos / divisor[:, 0::2])
        PE[:, 1::2] = torch.cos(pos / divisor[:, 1::2])
    
        # register
        self.register_buffer('PE', PE)

    def _build_model_architecture(self):
        """Build the model architecture (separated for reuse)"""
        # shared embedding
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab,
                                      embedding_dim=self.d,
                                      padding_idx=self.padding_idx)
        
        # positional embedding
        self.positional_encoding()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d=self.d, h=self.h, d_k=self.d_k, d_ff=self.d_ff, p_dropout=self.p_dropout) 
            for _ in range(self.N)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(h=self.h, d=self.d, d_k=self.d_k, d_ff=self.d_ff, p_dropout=self.p_dropout)
            for _ in range(self.N)
        ])
        
        # Output projection
        self.output_head = nn.Linear(self.d, self.n_vocab, bias=False)
        
        # Move to device
        self.to(self.device)

        # proper initialization of weights
        self._init_weights()

    def _load_checkpoint_weights(self, checkpoint_path):
        """Load weights from checkpoint file"""
        try:
            print(f"🔄 Loading weights from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
        except FileNotFoundError:
            print("Using random weights instead")
        except Exception as e:
            print("Using random weights instead")

    def _init_weights(self):
        """Initialize weights following Transformer paper guidelines"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # Standard LayerNorm initialization
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)