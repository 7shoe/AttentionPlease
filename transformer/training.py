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
                              L=64, 
                              d=args.d, 
                              h=8, 
                              d_ff=2048,
                              n_vocab=len(tokenizer.token_vocab), 
                              padding_idx=tokenizer.token_vocab['<PAD>'], 
                              bos_idx=tokenizer.token_vocab['<BOS>'], 
                              dtype=torch.float, 
                              device=device)
    
    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_vocab['<PAD>'])
    
    # optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    
    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
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
                ncols=120)

    # Training
    train_loss_list, val_loss_list = [], []
    train_time_list, val_time_list = [], []
    for epoch in range(1, args.n_epochs+1):
        transformer.train()
        total_loss = 0.0
    
        # batch
        for j,batch in enumerate(train_loader):
            src = batch['src_ids'].to(device)   # (B, S)
            tgt = batch['tgt_ids'].to(device)   # (B, T)
            
            # logits
            logits = transformer(src_ids=src, tgt_ids=tgt)
    
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

            # ??? val_loss.item() ??? 
            # val_loss_list.append(???)

            # progress update
            pbar.set_postfix({
                'Epoch': f'{epoch}/{args.n_epochs}',
                'Batch': f'{j+1}/{len(train_loader)}',
                'Epoch_Avg': f'{np.mean(train_loss_list[-10:]):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            pbar.update(1)
    
        # - end batch
    # - end of epoch

    # store model
    ckpt_dir = Path("/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transformer.state_dict(), ckpt_dir / f"{ts}_epoch{epoch}.pth")

    # log loss
    log_data = {
        'metadata': {
            'timestamp': ts,
            'n_epochs': args.n_epochs,
            'N': args.N,
            'learning_rate': args.learning_rate,
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
    
    # Add learning rate
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=1e-4,
                        help='Learning rate for optimizer (default: 1e-4)')
    
    # Optional: Add other useful arguments
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256,
                        help='Batch size for training (default: 256)')
    
    parser.add_argument('--d', 
                        type=int, 
                        default=512,
                        help='Model embedding dimension (default: 512)')

    # parse arguments
    args = parser.parse_args()
    
    # print config
    print("-" * 30)
    print("Training Configuration:")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Transformer layers (N): {args.N}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model dimension: {args.d}")
    print("-" * 30)

    main(args)