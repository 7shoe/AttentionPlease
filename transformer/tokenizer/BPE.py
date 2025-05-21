import re
import yaml
import numpy as np
import random
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

from utils.datasets import load_wmt
from utils.datasets import load_rnd

class Tokenizer:
    """
    Byte-Pair encoder (BPE): normalizes, tokenizes, one-hot encodes text
    """
    def __init__(self, 
                 compute_vocab:bool=False,
                 max_vocab_size:int = 100,
                 corpus_source:str='rnd',
                 vocab_dest_file:Path=Path('./data/dest/tokens.yaml'),
                 lower_case:bool=True,
                 special_tokens:list[str]=['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>', ' '],
                 count_threshold:int=5
                 ) -> None:
        
        self.token_vocab = dict()
        self.compute_vocab = compute_vocab
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        self.lower_case = lower_case
        self.count_threshold = 20

        # re-compute
        if self.compute_vocab:
            # load
            if corpus_source=='wmt':
                corpus = load_wmt()
            elif corpus_source=='rnd':
                corpus = load_rnd()
            else:
                raise NotImplementedError("`corpus_source` must be either `wmt` (WMT14, 100k train) or `rnd` (1000 random en sentences).")

            # corpus:str
            corpus = " ".join([str(sentence) for sentence in corpus])

            # normalize text
            self.corpus = self.normalize(text=corpus, lower_case=self.lower_case)

            # load initial vocab (merge table)
            self._initialize_vocab_(self.corpus.split())
            
            # merge
            self._update_bigram_table_()

            # output
            self.token_vocab = {str(token) : token_id for (token_id, token) in enumerate(self.token_vocab)}

            # store
            with open(vocab_dest_file, 'w') as f:
                yaml.dump(self.token_vocab, f)
        
        # load    
        else:  
            with open(vocab_dest_file, 'r') as f:
                self.token_vocab = yaml.safe_load(f)

        pass
            

    def _initialize_vocab_(self, 
                           list_of_words:list[str]) -> None:
        """
        Initialize single-character vocab (pre-merging)
        """
        
        # initialize w/ special tokens
        self.token_vocab = list(self.special_tokens)
        
        # word counter
        self.word_counter = Counter()
        for word in list_of_words:
            self.word_counter[' ' + word]+=1

        # initial length
        prev_len = len(self.word_counter)

        # remove rare words
        self.word_counter = Counter({word: cnt for word, cnt in self.word_counter.items() if cnt >= self.count_threshold})
        print(f'Eliminated {(prev_len-len(self.word_counter)) / (prev_len):.2f}% of rarest words from corpus for tokenization.')
        
        # read out char alphabet
        # - exclu. whitespace
        self.alphabet = sorted(set([char for word in self.word_counter for char in word if char!=' ']))
        # - incl. whitespace
        self.alphabet += sorted(set([word[:2] for word in self.word_counter]))
        
        # token vocabulary
        self.token_vocab += list(self.alphabet)

        # bigram token counter
        self.bigram_counter = Counter()
        
        # init bigram counter
        for c1 in self.alphabet:
            for c2 in self.alphabet:
                self.bigram_counter[(c1,c2)]+=sum([word_count * word.count(c1+c2) for (word, word_count) in self.word_counter.items() if (c1+c2 in word)])

        # (one-character) words that are already fully tokenized
        self.full_word_token_count = sum([1 for c in self.alphabet if c in self.word_counter])
    
        pass

    def _update_bigram_table_(self) -> None:
        """
        Conducts single merge-operation for given vocabulary (and given the frequencies)
        """

        # progress bar
        pbar = tqdm(total=self.max_vocab_size-len(self.token_vocab))
        
        # token vocabulary size limit not reached            
        while len(self.token_vocab) < self.max_vocab_size:
            # most frequent pair -> new token
            new_token, new_token_count = self.bigram_counter.most_common(1)[0]
            del self.bigram_counter[new_token]
            new_token = "".join(new_token)
            
            # - update bigram
            for c2 in self.alphabet:
                self.bigram_counter[(new_token,c2)]+=sum([word_count * word.count(new_token+c2) for (word, word_count) in self.word_counter.items() if (new_token+c2 in word)])
            
            # - add new token
            self.token_vocab.append(new_token)
        
            # new token = full word?
            if new_token in self.word_counter:
                self.full_word_token_count+=1
        
            # every word = token?
            if self.full_word_token_count >= len(self.word_counter):
                break

            # update 
            pbar.update(1)

        # close
        pbar.close()

        pass


    def normalize(self, text:str, lower_case:bool):
        """
        Filters symbols, excess escape-chars, etc. from input text
        """
        new_text = []
        if isinstance(text,str):
            was_type_str = True
            text = [text]
        else:
            was_type_str = False
        for text_segment in text:
            # filter non-character symbols
            text_segment = re.sub(r'[^a-zA-Z ]+', '', text_segment)
            
            # filter excess escape characters
            text_segment = re.sub(' +', ' ', text_segment)
    
            # lower-casing
            if lower_case:
                text_segment = text_segment.lower()
    
            new_text.append(text_segment)

        # revert to input type
        if was_type_str:
            new_text = " ".join(new_text)

        return new_text

    def tokenize(self, s:str) -> tuple[list[str],list[int]]:
        """
        Tokenize input text `s` according to token vocabulary
        """
        
        # inference
        s = self.normalize(s, self.lower_case)

        # destination
        token_seq = []
        i, j = 0, 1
        while i<len(s):
            # matched substring
            if s[i:i+j] in self.token_vocab:
                matched_token = s[i:i+j]
        
                # termination
                if i+j == len(s):
                    token_seq.append(matched_token)
                    break
        
                # increment
                j+=1
            # unknown first char
            elif j==1:
                token_seq.append('<UNK>')
                i+=1
        
            # too far: previous match & step-back
            else:
                token_seq.append(matched_token)
                i+=(j-1)
                j=1

        # encode token IDs
        token_id_seq = self.encode(token_seq)
        
        return (token_seq, token_id_seq)

    def encode(self, token_seq:list[str]) -> list[int]:
        """
        Given a token sequence, encode into sequence of token IDs
        (replace unknown tokens for robust inference)
        """
        
        token_id_seq = [self.token_vocab[token] if token in self.token_vocab else self.token_vocab['<UNK>'] for token in token_seq]

        return token_id_seq