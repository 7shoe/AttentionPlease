import argparse
from pathlib import Path
from tokenizer.BPE import Tokenizer

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=37_000, help="Max. size of token vocabulary.")
    parser.add_argument("-s", type=str, default='wmt', help="Source of text corpus.")
    parser.add_argument("-d", type=str, default='./data/dest/new_tokens.yaml', help="Source of text corpus.")
    parser.add_argument("--case_sensitive", action='store_true', help="Flag indicating case-sensitivity.")
    # parse
    args = parser.parse_args()
    
    # run BPE "training"
    t = Tokenizer(
        compute_vocab=True,
        max_vocab_size=args.n,
        corpus_source=args.s,
        vocab_dest_file=args.d,
        lower_case=not(args.case_sensitive),
        special_tokens=['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>']
    )

    #print('mommy help')

    pass


if __name__=='__main__':
    main()