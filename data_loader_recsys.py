import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]


        max_document_length = max([len(x.split(",")) for x in positive_examples])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping


    def build_vocab(self, sentences):
        vocab = {}
        ctr = 0
        for st in sentences:
            for ch in st:
                if ch not in vocab:
                    vocab[ch] = ctr
                    ctr += 1

        # SOME SPECIAL CHARACTERS
        vocab['eol'] = ctr
        vocab['padding'] = ctr + 1
        vocab['init'] = ctr + 2

        return vocab

    def string_to_indices(self, sentence, vocab):
        indices = [ vocab[s] for s in sentence.split(',') ]
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = { vocab[ch] : ch for ch in vocab } 
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]

        return "".join(sent)



