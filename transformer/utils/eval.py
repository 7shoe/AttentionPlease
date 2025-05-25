import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import argparse
from pathlib import Path
import time
import math
import numpy as np

def evaluate_model(model: nn.Module, 
                   val_loader: DataLoader, 
                   criterion: nn.Module, 
                   device: torch.device, 
                   max_batches: int = 50) -> float:
    """
    evaluate_model with detailed logging
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            # Forward pass
            logits = model(src_ids=src, tgt_ids=tgt)
            B, T, V = logits.shape
            
            # Calculate loss
            logits_reshaped = logits.view(B*T, V)
            tgt_reshaped = tgt.view(B*T)
            
            loss = criterion(logits_reshaped, tgt_reshaped)
            loss_item = loss.item()
            
            if math.isnan(loss_item):
                break
            
            total_loss += loss_item
            num_batches += 1
    
    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss