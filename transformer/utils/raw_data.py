import random
import pandas as pd
from pathlib import Path
import numpy as np

def load_wmt(corpus_source_file:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/wmt14/wmt14_translate_de-en_train.csv'),
             nrows:int=50_000,
             seed_val:int=678):
    """
    Loads subset of WMT14 EN<->DE dataset for BPE
    """
    np.random.seed(seed_val)
    rnd_idx_skips = sorted(list(set([int(num) for num in np.random.random(50_000)*100_000])))
    
    # load
    df = pd.read_csv(corpus_source_file, 
                     on_bad_lines='skip',
                     low_memory=False,
                     skiprows=rnd_idx_skips,
                     nrows = 2*nrows)
    
    # subset
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:nrows]

    # extract lists
    de_list = df['de'].tolist()
    en_list = df['en'].tolist()
    corpus = de_list + en_list
    
    # shuffle
    random.seed(seed_val)
    random.shuffle(corpus)

    print(f'len(corpus) : {len(corpus)}')
    
    return corpus

def load_wmt_chunk_df(subset:str='train',
                      corpus_source_dir:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/wmt14/'),
                      n_chunk:int=10_000,
                      start_idx:int=0):
    """
    Loads subset of WMT14 EN<->DE dataset in chunks of size `n_chunk` starting at index `start_idx`
    """

    assert subset in {'train', 'validation', 'test'}, "`subset` must be either `train`, `validation`, or `test`."
    
    # load file
    corpus_source_file = Path(corpus_source_dir) / f'wmt14_translate_de-en_{subset}.csv'
    assert corpus_source_file.is_file(), f"File path invalid: {corpus_source_file}"
    
    # load
    df = pd.read_csv(corpus_source_file, 
                     on_bad_lines='skip',
                     low_memory=False,
                     skiprows=range(1,start_idx),
                     nrows=n_chunk,
                     header=0)
    
    return df


def load_rnd(corpus_source_file:Path=Path('./data/source/random_english_sentences.txt')):
    """
    Load corpus of 1000 random sentences
    (Small, out-of-sample-encoder)
    """
    # load
    with open(corpus_source_file, 'r') as f:
        corpus = f.readlines()

    return corpus