import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class NeuralTranslationDataset(Dataset):
    def __init__(self,
                 subset:str,
                 dest_dir:Path=Path('/eagle/projects/argonne_tpc/siebenschuh/attention_from_scratch/data/preprocessed/wmt14_37k/'),
                 seq_len:int=64,
                 n_valid_rows:dict={'train' : 4_500_000, 'validation' : 3000, 'test' : 3000}):

        # dataset
        n_rows = n_valid_rows[subset]

        # open memmaps in r-only mode
        self.de = np.memmap(dest_dir / f'{subset}_de.dat', dtype=np.int32, mode='r', shape=(n_rows, seq_len))
        self.en = np.memmap(dest_dir / f'{subset}_en.dat', dtype=np.int32, mode='r', shape=(n_rows, seq_len))
        self.n_rows = n_rows
        self.seq_len = seq_len

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        # slice out one fixed-length row, convert to torch.Tensor
        src = torch.from_numpy(self.de[idx].copy()).long()
        tgt = torch.from_numpy(self.en[idx].copy()).long()
        
        return {'src_ids': src, 'tgt_ids': tgt}