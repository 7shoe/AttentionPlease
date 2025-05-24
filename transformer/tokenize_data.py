from pathlib import Path
from utils.preprocessing import tokenize_data

def main():
    # constants
    max_token_seq_len = 64
    chunksize = 2_000
    column_names = ['de', 'en']
    vocab_dest_file = Path('./data/dest/wmt_37k_tokens.yaml')
    dest_dir = Path('/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/preprocessed/wmt14_37k/')
    
    # loop subsets
    for subset in ['train', 'validation', 'test']:
        tokenize_data(subset=subset,
                      max_token_seq_len=max_token_seq_len,
                      chunksize=chunksize,
                      column_names=column_names,
                      vocab_dest_file=vocab_dest_file,
                      dest_dir=dest_dir)

    pass

if __name__=='__main__':
    main()