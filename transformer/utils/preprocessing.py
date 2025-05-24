from pathlib import Path
from tokenizer.BPE import Tokenizer
from utils.raw_data import load_wmt_chunk_df, get_wmt_df_len
import itertools
import pandas as pd
import numpy as np

def tokenize_data(subset:str,
                  max_token_seq_len:int=64,
                  chunksize:int=25_000,
                  column_names:list[str]=['de', 'en'],
                  vocab_dest_file:Path=Path('./data/dest/wmt_37k_tokens.yaml'),
                  dest_dir:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/preprocessed/wmt14_37k/')) -> None:
    """
    Tokenize the raw input data
    """
    assert subset in {'train', 'validation', 'test'}, "Pick one of those."
    assert dest_dir.is_dir(), f"Directory path invalid: {dest_dir}"
    assert vocab_dest_file.is_file(), f"Source file for tokenizer token vocabulary invalid: {vocab_dest_file}"

    # load tokenizer
    tokenizer = Tokenizer(compute_vocab=False, 
                          max_vocab_size=37_005,
                          corpus_source='wmt',
                          vocab_dest_file=Path('./data/dest/wmt_37k_tokens.yaml'))

    # number of rows
    n_rows = get_wmt_df_len(subset)
    
    # DEBUG
    print(f'n_rows : {n_rows}')
    
    # valid entries
    n_partitions = n_rows // chunksize + 1
    
    # 1st pass: valisd pairs
    valid_index = set(range(n_rows))
    for i in range(n_partitions):
        print(f'i={i}')
        # load DataFrame
        df = load_wmt_chunk_df(subset=subset, n_chunk=chunksize, start_idx=chunksize*i)
    
        # 1st pass: idenitfy valid sentence pairs
        for column in column_names:
            for j,sentence in enumerate(df[column].tolist()):
                if str(sentence)=='nan' and (j in valid_index):
                    valid_index.remove(j)

        del df
    
    # valid rows
    n_valid_rows = len(valid_index)
    print(f'n_valid_rows : {n_valid_rows}')
    
    # MEMmap
    # - German
    mm_de = np.memmap(dest_dir / f'{subset}_de.dat',
                      dtype=np.int32,
                      mode="w+",
                      shape=(n_valid_rows, max_token_seq_len))
    # - English
    mm_en = np.memmap(dest_dir / f'{subset}_en.dat',
                      dtype=np.int32,
                      mode="w+",
                      shape=(n_valid_rows, max_token_seq_len))
    
    # pre-fill
    mm_de[:] = tokenizer.token_vocab['<PAD>']
    mm_de.flush()
    mm_en[:] = tokenizer.token_vocab['<PAD>']
    mm_en.flush()
    
    # create MEMmap
    for i in range(n_partitions):
        # load DataFrame
        df = load_wmt_chunk_df(subset=subset, n_chunk=chunksize, start_idx=chunksize*i)
    
        # 1st pass: idenitfy valid sentence pairs
        valid_index = set(range(len(df)))
        for column in column_names:
            for j,sentence in enumerate(df[column].tolist()):
                if str(sentence)=='nan':
                    valid_index.remove(j)
        
        # 2nd pass: tokenize
        for k,column in enumerate(column_names):
            for j,sentence in enumerate(df[column].tolist()):
                if j in valid_index:
                    tokens,token_ids = tokenizer.tokenize(str(sentence))
    
                    # pad/truncate token_ids seq exactly to `max_token_seq_len` 
                    if len(token_ids) < max_token_seq_len-2:
                        seq = [tokenizer.token_vocab['<BOS>']] + token_ids + [tokenizer.token_vocab['<EOS>']] + [tokenizer.token_vocab['<PAD>']] * (max_token_seq_len-len(token_ids)-2)
                    else:
                        seq = [tokenizer.token_vocab['<BOS>']] + token_ids[:max_token_seq_len-2] + [tokenizer.token_vocab['<EOS>']]
    
                    # append
                    if column=='de':
                        mm_de[chunksize*i + j, :] = seq
                    else:
                        mm_en[chunksize*i + j, :] = seq
        
    # write to disk
    mm_de.flush()
    del mm_de 
    mm_en.flush()
    del mm_en

    # Done
    pass