import ptb_data_generator as dat_gen
import numpy as np


def create_rev_dict(filename):
    words_to_ids = dat_gen._build_vocab(filename)
    ids_to_words = {v: k for k, v in words_to_ids.items()}
    return ids_to_words


if __name__ == '__main__':
    ids_to_words = create_rev_dict("/Users/thomasklein/Uni/Bachelorarbeit/ptbtext/ptb.train.txt")
    words_to_ids = dat_gen._build_vocab("/Users/thomasklein/Uni/Bachelorarbeit/ptbtext/ptb.train.txt")
    print(ids_to_words[words_to_ids['<eos>']])