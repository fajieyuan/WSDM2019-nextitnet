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





    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text)%sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed


    def load_translation_data(self):
        source_lines = []
        target_lines = []
        for i in range(len(self.source_lines)):
            source_lines.append( self.string_to_indices(self.source_lines[i], self.source_vocab) )
            target_lines.append( self.string_to_indices(self.target_lines[i], self.target_vocab) )

        buckets = self.create_buckets(source_lines, target_lines)

        
        return buckets, self.source_vocab, self.target_vocab


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

   
