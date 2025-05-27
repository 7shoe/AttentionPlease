from pathlib import Path
from tokenizer.BPE import Tokenizer
from utils.raw_data import load_wmt_chunk_df, get_wmt_df_len
import numpy as np
import math
from typing import Optional
import yaml
import argparse
import time
from tqdm import tqdm
from utils.logging import save_yaml_log
from utils.eval import evaluate_model

import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from model import Transformer

from torch.utils.data import DataLoader
from utils.dataset import NeuralTranslationDataset


def get_transformer_schedule(warmup_steps: int = 4000):
    """
    Transformer learning rate schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Square root decay after warmup
    """
    def lr_lambda(step):
        step = max(1, step)  # Avoid division by zero
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Square root decay
            return (warmup_steps / step) ** 0.5
    return lr_lambda

def main(args):
    
    # start time
    ts = round(time.time())
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # load tokenizer
    tokenizer = Tokenizer(compute_vocab=False, 
                          max_vocab_size=37_005,
                          corpus_source='wmt',
                          vocab_dest_file=Path('./data/dest/wmt_37k_tokens.yaml'))

    # device
    device = torch.device('cuda')
    
    # model
    transformer = Transformer(N=args.N, 
                              L_max=args.L_max, 
                              d=args.d, 
                              h=args.h, 
                              d_ff=args.d_ff,
                              p_dropout=args.p_dropout,
                              n_vocab=len(tokenizer.token_vocab), 
                              padding_idx=tokenizer.token_vocab['<PAD>'], 
                              bos_idx=tokenizer.token_vocab['<BOS>'], 
                              dtype=torch.float, 
                              device=device)
    
    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_vocab['<PAD>'], label_smoothing=args.label_smoothing)
    
    # optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    
    # scheduler
    if args.pre_ln:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    else:
        warmup_steps = 4
        scheduler = lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=get_transformer_schedule(warmup_steps=warmup_steps)
        )
        
    # dataset
    data_train = NeuralTranslationDataset(subset='train')        # TODO -> `train`
    data_val   = NeuralTranslationDataset(subset='validation')
    
    # dataloaders
    # -train
    train_loader = DataLoader(data_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4, 
                              pin_memory=True)
    # -val
    val_loader = DataLoader(data_val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4, 
                            pin_memory=True)

    # validation requency
    val_frequency = max(1, len(train_loader) // 1000) # ~1%
    
    # Calculate total iterations for progress bar
    total_iterations = args.n_epochs * len(train_loader)
    
    # Create single progress bar for entire training
    pbar = tqdm(total=total_iterations, 
                desc="Training Progress",
                unit="batch",
                ncols=150)

    # logging data
    global_step = 0
    train_loss_list, val_loss_list = [], []
    train_time_list, val_time_list = [], []

    # Training
    for epoch in range(1, args.n_epochs+1):
        transformer.train()
        total_loss = 0.0
    
        # batch
        for j,batch in enumerate(train_loader):
            src = batch['src_ids'].to(device)   # (B, S)
            tgt = batch['tgt_ids'].to(device)   # (B, T)
            
            # logits
            logits = transformer(src_ids=src, tgt_ids=tgt)

            # 
            if j%150==0:
                print(f'logits.size(): {logits.size()}')
    
            B, T, V = logits.shape
    
            # loss
            loss = criterion(logits.view(B*T, V), tgt.view(B*T))
            
            # d) backward + step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()

            # track loss
            total_loss += loss.item()
            train_loss_list.append(loss.item())
            train_time_list.append(time.time())

            # validation loss
            if j % val_frequency == 0 and j > 0:
                val_loss = evaluate_model(transformer, val_loader, criterion, device, max_batches=50)
                val_loss_list.append(val_loss)
                val_time_list.append(time.time())

            # meta data
            scheduler.step()
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            
            # progress update
            pbar.set_postfix({
                'Epoch': f'{epoch}/{args.n_epochs}',
                'Batch': f'{j+1}/{len(train_loader)}',
                'Train_loss': f'{np.mean(train_loss_list[-10:]):.4f}',
                'LR': f'{current_lr:.2e}',
                'Step': global_step,
            })
            pbar.update(1)
    
        # - end batch
    # - end of epoch

    # store model
    ckpt_dir = Path("/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file_path = ckpt_dir / f"{ts}_epoch{epoch}.pth"
    torch.save(transformer.state_dict(), ckpt_file_path)

    # print checkpoint path
    print(f'Checkpoint: {ckpt_file_path}')

        #n_vocab=len(tokenizer.token_vocab), 
                              #padding_idx=tokenizer.token_vocab['<PAD>'], 
                              #bos_idx=tokenizer.token_vocab['<BOS>'], 
                              #dtype=torch.float, 

    # args dictionary
    args_dict = vars(args) | {'n_vocab' : len(tokenizer.token_vocab), 
                              'padding_idx' : tokenizer.token_vocab['<PAD>'], 
                              'bos_idx' : tokenizer.token_vocab['<BOS>']}

    # log loss
    log_data = {
        'args': args_dict,
        'metadata': {
            'timestamp': ts,
            'n_epochs': args.n_epochs,
            'N': args.N,
            'learning_rate': args.learning_rate,
            'checkpoint': str(ckpt_file_path)
        },
        'training_log': {
            'train_losses': train_loss_list,
            'val_losses': val_loss_list,
            'train_times': train_time_list,
            'val_times': val_time_list,
        }
    }
    
    # - save
    success = save_yaml_log(f'./logs/{ts}.yaml', log_data)

    pass

if __name__=='__main__':
    # parser
    parser = argparse.ArgumentParser(description='Train Transformer for Neural Machine Translation')

    # Add n_epochs int no default (required)
    parser.add_argument('--n_epochs', 
                        type=int, 
                        required=True,
                        help='Number of training epochs (required)')
    
    # Add N int, default is 6
    parser.add_argument('--N', 
                        type=int, 
                        default=6,
                        help='Number of transformer layers (default: 6)')
    
    # learning rate
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=1e-4,
                        help='Learning rate for optimizer (default: 1e-4)')

    # label smoothing
    parser.add_argument('--label_smoothing', 
                        type=float, 
                        default=0.0,
                        help='Label smoothing (default: 0.0; i.e. none).')
    
    # batch size
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256,
                        help='Batch size for training (default: 256)')

    # embedding dimension
    parser.add_argument('--d', 
                        type=int, 
                        default=512,
                        help='Model embedding dimension (default: 512)')

    # number of heads
    parser.add_argument('--h', 
                        type=int, 
                        default=8,
                        help='Number of heads (default: 8)')

    # L_max
    parser.add_argument('--L_max', 
                        type=int, 
                        default=64,
                        help='Maximum number of sequence tokens (default: 64)')  # d_ff

    # d_ff
    parser.add_argument('--d_ff', 
                        type=int, 
                        default=2048,
                        help='FFN embedding dimension (default: 2048)')

    # dropout
    parser.add_argument('--p_dropout', 
                        type=float,
                        default=0.0,
                        help='Dropout probability (default: 0.1)')
    
    # pre-layer normalization flag
    parser.add_argument('--pre_ln', 
                        action='store_true',
                        help='Use pre-layer normalization instead of post-layer normalization (default: False)')

    # parse arguments
    args = parser.parse_args()

    # print config
    print("-" * 30)
    print("Training Configuration:")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Transformer layers (N): {args.N}")
    print(f"  Learning rate (lr): {args.learning_rate}")
    print(f"  Batch size (B): {args.batch_size}")
    print(f"  Hidden dimension (d_model): {args.d}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Dropout probability (p_dropout): {args.p_dropout}")
    print(f"  Pre-layer norm: {args.pre_ln}")
    print("-" * 30)

    main(args)