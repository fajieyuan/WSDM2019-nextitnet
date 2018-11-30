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



    def create_buckets(self, source_lines, target_lines):
        
        bucket_quant = self.bucket_quant
        source_vocab = self.source_vocab
        target_vocab = self.target_vocab

        buckets = {}
        for i in xrange(len(source_lines)):
            
            source_lines[i] = np.concatenate( (source_lines[i], [source_vocab['eol']]) )
            target_lines[i] = np.concatenate( ([target_vocab['init']], target_lines[i], [target_vocab['eol']]) )
            
            sl = len(source_lines[i])
            tl = len(target_lines[i])


            new_length = max(sl, tl)
            if new_length % bucket_quant > 0:
                new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant    
            
            s_padding = np.array( [source_vocab['padding'] for ctr in xrange(sl, new_length) ] )

            # NEED EXTRA PADDING FOR TRAINING.. 
            t_padding = np.array( [target_vocab['padding'] for ctr in xrange(tl, new_length + 1) ] )

            source_lines[i] = np.concatenate( [ source_lines[i], s_padding ] )
            target_lines[i] = np.concatenate( [ target_lines[i], t_padding ] )

            if new_length in buckets:
                buckets[new_length].append( (source_lines[i], target_lines[i]) )
            else:
                buckets[new_length] = [(source_lines[i], target_lines[i])]

            if i%1000 == 0:
                print "Loading", i
            
        return buckets

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

    def get_batch_from_pairs(self, pair_list):
        source_sentences = []
        target_sentences = []
        for s, t in pair_list:
            source_sentences.append(s)
            target_sentences.append(t)

        return np.array(source_sentences, dtype = 'int32'), np.array(target_sentences, dtype = 'int32')


