import os
import json
import random
import numpy as np
import tensorflow as tf

import sys

from repo.official.nlp.bert import tokenization

UNKNOWN_WORD = '<UNKNONW>'
UNKNOWN_TAG = '<UNKNOWN_TAG>'
PADDING = '<PADDING>'
WORD_IDX=0
TAG_IDX=1
SEP='|'

CHAR_SIZE=10

def copy_vocabulary(src_gen, dst_gen):
    dst_gen.word2idx = src_gen.word2idx
    dst_gen.tag2idx = src_gen.tag2idx
    dst_gen.char2idx = src_gen.char2idx
    dst_gen.tags = src_gen.tags

def pred2label(pred, tag2idx):
        out = []
        idx2tag = {idx : tag for tag, idx in tag2idx.items()}
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(idx2tag[p_i])
            out.append(out_i)
        return out[0]

class DataGeneratorLSTMChars(tf.keras.utils.Sequence):
    def __init__(
        self,
        base_path=".",
        path_to_data = "",
        fit = True,
        batch_size = 16,
        max_seq_length = 64,
        max_char_len=10,
        shuffle = True,
    ):
        self.batch_size = batch_size  #batch size
        self.shuffle = shuffle  #Whether to shuffle the data
        self.fit = fit

        self.path_to_data = base_path + '/' + path_to_data
        self.max_seq_length = max_seq_length
        self.max_char_length = max_char_len
        self.data = []
        self._read_data()
        self._get_vocabulary()
                
    def _read_data(self):
        
        with open(self.path_to_data, "r", encoding="utf8") as input_file:
            curr_words = []
            curr_tags = []
            for i, line in enumerate(input_file):
                
                line = line.rstrip()
                if len(line) == 0:
                    self.data.append((curr_words, curr_tags))
                    curr_words = []
                    curr_tags = []
                else:
                    entry = tuple(line.split(SEP))
                    try:
                        word = entry[WORD_IDX]
                        tag = entry[TAG_IDX]
                        curr_words.append(word)
                        curr_tags.append(tag)
                    except:
                        print('line ', i, ' : ', line)
                        print('entry: ', entry)   

    def _get_vocabulary(self):
        self.words = set()
        self.tags = set() 
        self.chars = set()

        self.word2idx = {}
        self.tag2idx = {}

        self.words.add(UNKNOWN_WORD)
        self.words.add(PADDING)
        self.tags.add(UNKNOWN_TAG)
        self.tags.add(PADDING)

        for entry in self.data:
            self.words = self.words.union(entry[0])
            for w in entry[0]:
                for c in w:
                    self.chars.add(c)
            
            self.tags = self.tags.union(entry[1])
    
        self.words = sorted(list(self.words))
        self.tags = sorted(list(self.tags))

        self.word2idx = { w : i for i, w in enumerate(self.words) }
        self.char2idx = { c : i + 1 for i, c in enumerate(self.chars)}
        self.tag2idx  = { t : i for i, t in enumerate(self.tags) }
        self.idx2tag = {i : w for w, i in self.tag2idx.items()}

    def __len__(self):
        return len(self.data) // self.batch_size

    def _sentence2idxs(self, sentence):
        sentence_idxs = list(map(lambda w: self.word2idx.get(w, self.word2idx[UNKNOWN_WORD]), sentence))
        return sentence_idxs

    def _sentence_tags2idxs(self, sentence):
        sentence_tag_idxs = list(map(lambda t: self.tag2idx.get(t, self.tag2idx[UNKNOWN_TAG]), sentence))    
        return sentence_tag_idxs

    def _word2charidxs(self, word):
        char_feats = list(map(lambda c : self.char2idx.get(c, 0), word))
        return char_feats

    def _get_char_features(self, X):
        X_chars = []
        for sent in X:
            sent_indx = list(map(self._word2charidxs, sent))
            sent_indx = tf.keras.preprocessing.sequence.pad_sequences(maxlen=self.max_char_length,
             sequences=sent_indx, padding="post", truncating="post", value=0)
            X_chars.append(sent_indx)
        pad_val = np.zeros((self.max_seq_length, self.max_char_length))
        X_chars = tf.keras.preprocessing.sequence.pad_sequences(maxlen=self.max_seq_length, sequences=X_chars,
         padding="post", truncating="post", value = pad_val)
        return X_chars

    def __getitem__(self, index):
        entries = self.data[index*self.batch_size: (index+1)*self.batch_size]

        X_all = []
        y_all = []
        for entry in entries:
            X = self._sentence2idxs(entry[0])[:self.max_seq_length]
            X = X + [self.word2idx[PADDING]] * (self.max_seq_length - len(X))

            y = self._sentence_tags2idxs(entry[1])[:self.max_seq_length]
            y = y + [self.tag2idx[PADDING]] * (self.max_seq_length - len(y))

            y = tf.keras.utils.to_categorical(y, num_classes=len(self.tags) + 1)

            X_all.append(X)
            y_all.append(y)

        X_chars_all = self._get_char_features(list(map(lambda entry: entry[0], entries)))

        return [np.array(X_all), np.array(X_chars_all)], np.array(y_all)

class DataGeneratorLSTMCharsCRF(DataGeneratorLSTMChars):
    def __init__(self, 
        base_path='.', 
        path_to_data='', 
        fit=True, 
        batch_size=16, 
        max_seq_length=64, 
        max_char_len=10, 
        shuffle=True):
        super().__init__(base_path=base_path, path_to_data=path_to_data, fit=fit, batch_size=batch_size, max_seq_length=max_seq_length, max_char_len=max_char_len, shuffle=shuffle)

    def __getitem__(self, index):
        entries = self.data[index*self.batch_size: (index+1)*self.batch_size]

        X_all = []
        y_all = []
        for entry in entries:
            X = self._sentence2idxs(entry[0])[:self.max_seq_length]
            X = X + [self.word2idx[PADDING]] * (self.max_seq_length - len(X))

            y = self._sentence_tags2idxs(entry[1])[:self.max_seq_length]
            y = y + [self.tag2idx[PADDING]] * (self.max_seq_length - len(y))

            X_all.append(X)
            y_all.append(y)

        X_chars_all = self._get_char_features(list(map(lambda entry: entry[0], entries)))

        return [np.array(X_all), np.array(X_chars_all)], np.array(y_all)

class DataGeneratorBert(tf.keras.utils.Sequence):
    '''
    This is a data generator class
    '''
    def __init__(
        self,
        base_path=".",
        path_to_data = "",
        path_to_bert = "BERT",
        fit = True,
        batch_size = 32,
        max_seq_length = 128,
        shuffle = True,
    ):
        path_to_bert = base_path + "/" + path_to_bert
        path_to_data = base_path + "/" + path_to_data
        #initializing the tokenizer
        self.tokenizer =  tokenization.FullTokenizer(vocab_file=os.path.join(path_to_bert, \
            "assets/vocab.txt"), do_lower_case=True)
        self.path_to_data = path_to_data 
        self.data = []
        
        

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fit = fit

        self.input_tokens = []

        self.input_ids = []
        self.input_mask = []
        self.segment_ids = []

        self.input_tags = []

        self._read_data()
        self._get_vocabulary()

        for index, data_point in enumerate(self.data):
            input_tokens = []
            input_tags = []
            
            for i, word in enumerate(data_point[0]):
                toks = self.tokenizer.tokenize(word)
                tags = [data_point[1][i]]
                if len(toks) > 1:
                    if tags[0].startswith('B-') or tags[0].startswith('I-'):
                        complete = 'I-' + ''.join(tags[0][2:])
                    else:
                        complete = 'O'
                    n_add = len(toks) - len(tags)
                    tags = tags + [complete] * n_add


                input_tokens.extend(toks)
                input_tags.extend(tags)
                assert len(input_tokens) == len(input_tags), print(input_tokens, input_tags)

            input_tokens = ['[CLS]'] + input_tokens + ['[SEP]']

            self.input_tokens.append(input_tokens)
            
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            input_ids += [0]*(self.max_seq_length - len(input_ids))
            
            input_mask = [1] * len(input_ids)
            input_mask += [0]*(self.max_seq_length - len(input_mask))

            segment_ids = [1] * (len(input_ids) + 1)
            segment_ids += [0]*(self.max_seq_length - len(segment_ids))

            self.input_ids.append(input_ids[:self.max_seq_length])
            self.input_mask.append(input_mask[:self.max_seq_length])
            self.segment_ids.append(input_mask[:self.max_seq_length])

            input_tags = input_tags + [PADDING] * (self.max_seq_length - len(input_tags))

            self.input_tags.append(input_tags[:self.max_seq_length])

    def _read_data(self):
        with open(self.path_to_data, "r", encoding="utf8") as input_file:
            curr_words = []
            curr_tags = []
            for i, line in enumerate(input_file):
                
                line = line.rstrip()
                if len(line) == 0:
                    self.data.append((curr_words, curr_tags))
                    curr_words = []
                    curr_tags = []
                else:
                    entry = tuple(line.split(SEP))
                    try:
                        word = entry[WORD_IDX]
                        tag = entry[TAG_IDX]
                        curr_words.append(word)
                        curr_tags.append(tag)
                    except:
                        print('line ', i, ' : ', line)
                        print('entry: ', entry)  

    def _get_vocabulary(self):
        self.tags = set() 
        self.tag2idx = {}

        self.tags.add(UNKNOWN_TAG)
        self.tags.add(PADDING)

        for entry in self.data:  
            self.tags = self.tags.union(entry[1])
    
        self.tags = sorted(list(self.tags))
        self.tag2idx  = { t : i for i, t in enumerate(self.tags) }

    def _sentence_tags2idxs(self, sentence):
        sentence_tag_idxs = list(map(lambda t: self.tag2idx.get(t, self.tag2idx[UNKNOWN_TAG]), sentence))    
        return sentence_tag_idxs

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):

        input_ids = np.array(self.input_ids[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        input_mask = np.array(self.input_mask[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        segment_ids = np.array(self.segment_ids[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")

        input_tags_ = self.input_tags[index*self.batch_size: (index+1)*self.batch_size]
        
        input_tags = []
        for tags in input_tags_:
            tags = self._sentence_tags2idxs(tags)
            tags = tf.keras.utils.to_categorical(tags, num_classes=len(self.tags) + 1)
            tags = np.array(tags)
            input_tags.append(tags)
        
        input_list = [input_ids, input_mask, segment_ids]
        return input_list, np.array(input_tags)

class DataGeneratorBertCRF(DataGeneratorBert):
    def __init__(self, base_path='.', path_to_data='', path_to_bert='BERT', fit=True, batch_size=32,
     max_seq_length=128, shuffle=True):
        super().__init__(base_path=base_path, path_to_data=path_to_data, path_to_bert=path_to_bert,
         fit=fit, batch_size=batch_size, max_seq_length=max_seq_length, shuffle=shuffle)
    
    def __getitem__(self, index):

        input_ids = np.array(self.input_ids[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        input_mask = np.array(self.input_mask[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        segment_ids = np.array(self.segment_ids[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")

        input_tags_ = self.input_tags[index*self.batch_size: (index+1)*self.batch_size]
        
        input_tags = []
        for tags in input_tags_:
            tags = self._sentence_tags2idxs(tags)
            tags = np.array(tags)
            input_tags.append(tags)
        
        input_list = [input_ids, input_mask, segment_ids]
        return input_list, np.array(input_tags)