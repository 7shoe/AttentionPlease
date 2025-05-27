# generator.py - Standalone Text Generation for Transformer
import torch
import torch.nn.functional as F
from typing import Optional, Union, List

class GeneratorLayer:
    """
    Standalone text generation layer for Transformer encoder-decoder models.
    Handles autoregressive decoding with various strategies.
    """
    
    def __init__(self, model, max_length: int = 100):
        """
        Args:
            model: Transformer model instance
            max_length: Maximum generation length
        """
        self.model = model
        self.max_length = max_length
        
    @torch.no_grad()
    def generate(self, 
                 src_ids: torch.LongTensor,
                 L_max: int = 64,
                 temperature: float = 1.0,
                 eos_idx: int = 2,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 do_sample: bool = True) -> torch.LongTensor:
        """
        Generate text autoregressively
        
        Args:
            src_ids: Source token IDs [batch_size, src_len]
            L_max: Maximum target sequence length
            eos_idx: End-of-sequence token ID
            temperature: Sampling temperature (1.0 = no change, <1.0 = more focused)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold  
            do_sample: If False, use greedy decoding
            
        Returns:
            Generated token IDs [batch_size, generated_len]
        """
        self.model.eval()
        device = src_ids.device
        batch_size = src_ids.size(0)
        
        # Encode source sequence once
        encoded = self._encode_source(src_ids)
        
        # Initialize target sequence with BOS token
        tgt_ids = torch.full((batch_size, 1), self.model.bos_idx, 
                           dtype=torch.long, device=device)
        
        # Generate tokens autoregressively
        for step in range(L_max - 1): 
            # next token-logits
            logits = self._decode_step(encoded, tgt_ids, src_ids)
            next_token_logits = logits[:, -1, :]  # Last position
            
            # apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # top-k filtering
            if top_k is not None:
                next_token_logits = self._top_k_filter(next_token_logits, top_k)
            
            # apply top-p (nucleus) filtering
            if top_p is not None:
                next_token_logits = self._top_p_filter(next_token_logits, top_p)
            
            # sample next token
            if do_sample and temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # append to sequence
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
            
            # Check for EOS token (early stopping)
            if (next_token == eos_idx).all():
                break
        
        return tgt_ids
    
    def _encode_source(self, src_ids: torch.LongTensor) -> torch.Tensor:
        """Encode source sequence"""
        B, L_src = src_ids.size()
        
        # Embedding + positional encoding
        X_enc = self.model.embedding(src_ids)
        X_enc = X_enc * torch.sqrt(torch.tensor(self.model.d, dtype=X_enc.dtype)) + self.model.PE[:L_src]
        
        # Source mask
        src_pad_mask = src_ids.eq(self.model.padding_idx)
        src_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)
        
        # Encode
        for encoder_layer in self.model.encoder_layers:
            X_enc = encoder_layer(X_enc, mask=src_mask)
        
        return X_enc
    
    def _decode_step(self, encoded: torch.Tensor, tgt_ids: torch.LongTensor, 
                    src_ids: torch.LongTensor) -> torch.Tensor:
        """Single decoding step"""
        B, L_tgt = tgt_ids.size()
        
        # Target embedding + positional encoding
        X_dec = self.model.embedding(tgt_ids)
        X_dec = X_dec * torch.sqrt(torch.tensor(self.model.d, dtype=X_dec.dtype)) + self.model.PE[:L_tgt]
        
        # Target self-attention mask (causal + padding)
        tgt_pad_mask = tgt_ids.eq(self.model.padding_idx)
        causal = self.model.causal_mask(L_tgt)
        causal = causal.unsqueeze(0).unsqueeze(0)
        tgt_key_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = causal | tgt_key_mask
        
        # Cross-attention mask (source padding)
        cross_attn_mask = src_ids.eq(self.model.padding_idx).unsqueeze(1).unsqueeze(1)
        
        # Decode
        for decoder_layer in self.model.decoder_layers:
            X_dec = decoder_layer(X_dec, encoded, 
                                self_attn_mask=tgt_mask,
                                cross_attn_mask=cross_attn_mask)
        
        # Project to vocabulary
        logits = self.model.output_head(X_dec)
        return logits
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k logits, set others to -inf"""
        if k <= 0:
            return logits
        
        top_k_logits, _ = torch.topk(logits, k, dim=-1)
        min_values = top_k_logits[:, -1].unsqueeze(-1)
        return torch.where(logits >= min_values, logits, 
                          torch.full_like(logits, float('-inf')))
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling: keep tokens with cumulative probability <= p"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def beam_search(self, 
                   src_ids: torch.LongTensor,
                   L_max: int = 64,
                   eos_idx: int = 2,
                   beam_size: int = 4) -> List[torch.LongTensor]:
        """
        Simple beam search decoding (optional - for better quality)
        
        Returns:
            List of generated sequences, sorted by score
        """
        self.model.eval()
        device = src_ids.device
        batch_size = src_ids.size(0)
        
        if batch_size != 1:
            raise NotImplementedError("Beam search currently supports batch_size=1 only")
        
        # Encode source
        encoded = self._encode_source(src_ids)
        
        # Initialize beam with BOS token
        sequences = [torch.full((1, 1), self.model.bos_idx, dtype=torch.long, device=device)]
        scores = [0.0]
        
        for step in range(L_max - 1):
            candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[0, -1].item() == eos_idx:
                    # Sequence already ended
                    candidates.append((scores[i], seq))
                    continue
                
                # Get logits for this sequence
                logits = self._decode_step(encoded, seq, src_ids)
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top beam_size tokens
                top_log_probs, top_tokens = torch.topk(log_probs, beam_size)
                
                for j in range(beam_size):
                    new_seq = torch.cat([seq, top_tokens[j:j+1].unsqueeze(0)], dim=1)
                    new_score = scores[i] + top_log_probs[j].item()
                    candidates.append((new_score, new_seq))
            
            # Keep top beam_size candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:beam_size]
            
            sequences = [seq for _, seq in candidates]
            scores = [score for score, _ in candidates]
            
            # Early stopping if all sequences ended
            if all(seq[0, -1].item() == eos_idx for seq in sequences):
                break
        
        return sequences

def decode_tokens(token_ids: torch.LongTensor, tokenizer, skip_special_tokens: bool = True) -> List[str]:
    """
    Decode token IDs back to text
    
    Args:
        token_ids: Generated token IDs [batch_size, seq_len]
        tokenizer: Your tokenizer instance
        skip_special_tokens: Whether to skip special tokens like BOS, EOS, PAD
        
    Returns:
        List of decoded strings
    """
    results = []
    for seq in token_ids:
        if skip_special_tokens:
            # Remove special tokens
            seq = seq.cpu().tolist()
            special_tokens = {tokenizer.token_vocab.get('<PAD>', -1), 
                            tokenizer.token_vocab.get('<BOS>', -1),
                            tokenizer.token_vocab.get('<EOS>', -1)}
            seq = [token for token in seq if token not in special_tokens]
        else:
            seq = seq.cpu().tolist()
        
        # Decode using your tokenizer
        text = tokenizer.decode(seq)  # Adjust this based on your tokenizer API
        results.append(text)
    
    return results

def generate_with_context(model, tokenizer, source_text: str, **generation_kwargs) -> str:
    """
    High-level generation function
    
    Args:
        model: Transformer model
        tokenizer: Your tokenizer
        source_text: Input text to translate/continue
        **generation_kwargs: Arguments for generate()
        
    Returns:
        Generated text
    """
    # Encode source text
    src_tokens = tokenizer.encode(source_text)
    src_ids = torch.LongTensor([src_tokens]).to(model.device)
    
    # Generate
    generator = GeneratorLayer(model)
    generated_ids = generator.generate(src_ids, 
                                     eos_idx=tokenizer.token_vocab.get('<EOS>', 2),
                                     **generation_kwargs)
    
    # Decode
    generated_texts = decode_tokens(generated_ids, tokenizer)
    return generated_texts[0]