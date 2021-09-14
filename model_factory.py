import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from tensorflow_addons.layers import CRF

import sys

from repo.official.nlp.bert import tokenization

import crf

def build_LSTM_Chars(max_seq_len, max_char_len, n_words, n_chars, n_tags):
    # word embedding
    word_input = keras.layers.Input(shape=(max_seq_len,))
    word_emb = keras.layers.Embedding(input_dim=n_words + 1, output_dim=50,
                    input_length=max_seq_len, mask_zero=False,
                    )(word_input)
    #               trainable=False, weights=[embeddings])(word_input)
    word_feats = keras.layers.Dropout(0.5)(word_emb)
    # char encoding
    char_input = keras.layers.Input(shape=(max_seq_len, max_char_len))
    char_emb = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=n_chars + 1,
        output_dim=30, input_length=max_char_len))(char_input)  

    char_dropout = keras.layers.Dropout(0.5)(char_emb)
    char_conv1d = keras.layers.TimeDistributed(keras.layers.Conv1D(kernel_size=3, filters=32,
        padding='same',activation='tanh', strides=1))(char_dropout)
    char_maxpool = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(max_char_len))(char_conv1d)
    char_feats = keras.layers.TimeDistributed(keras.layers.Flatten())(char_maxpool)

    all_feat = keras.layers.concatenate([word_feats, char_feats])
    all_out = keras.layers.SpatialDropout1D(0.3)(all_feat)

    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=max_seq_len,
            return_sequences=True))(all_out)

    out = keras.layers.TimeDistributed(keras.layers.Dense(n_tags + 1,
    activation="softmax"))(bi_lstm)

    model = keras.models.Model([word_input, char_input], out)

    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy)
    
    return model

def build_LSTM_Chars_CRF(max_seq_len, max_char_len, n_words, n_chars, n_tags):
    # word embedding
    word_input = keras.layers.Input(shape=(max_seq_len,))
    word_emb = keras.layers.Embedding(input_dim=n_words + 1, output_dim=50,
                    input_length=max_seq_len, mask_zero=False,
                    )(word_input)
    #               trainable=False, weights=[embeddings])(word_input)
    word_feats = keras.layers.Dropout(0.5)(word_emb)
    # char encoding
    char_input = keras.layers.Input(shape=(max_seq_len, max_char_len))
    char_emb = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=n_chars + 1,
        output_dim=30, input_length=max_char_len))(char_input)  

    char_dropout = keras.layers.Dropout(0.5)(char_emb)
    char_conv1d = keras.layers.TimeDistributed(keras.layers.Conv1D(kernel_size=3, filters=32,
        padding='same',activation='tanh', strides=1))(char_dropout)
    char_maxpool = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(max_char_len))(char_conv1d)
    char_feats = keras.layers.TimeDistributed(keras.layers.Flatten())(char_maxpool)

    all_feat = keras.layers.concatenate([word_feats, char_feats])
    all_out = keras.layers.SpatialDropout1D(0.3)(all_feat)

    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=max_seq_len,
            return_sequences=True))(all_out)
    crf_layer = CRF(units=n_tags+1)
    
    out = crf_layer(bi_lstm)

    model = keras.models.Model([word_input, char_input], out)

    model = crf.ModelWithCRFLoss(model)

    model.compile(optimizer="adam")
    
    return model

class BertLayer(tf.keras.layers.Layer):
    '''
      This is the bert layer subclass
    '''
    def __init__(
        self,
        path_to_bert = r"D:\\Facultate\\Master\\Disertatie_V2\\BERT",
        trainable = False,
        pooling = "mean",
        num_fine_tuning_layers = 12,
    ):
        super().__init__()
        self.trainable = trainable #This handles whether you want to train bert layer or not
        self.num_fine_tuning_layers = num_fine_tuning_layers #If you are training , then specify how many layers to fine tune
        self.path_to_bert = path_to_bert #specify the path to bert pretrained model
        self.pooling = pooling #if true, returns only the [CLS] token embedding

    '''
    This function is responsible to initialize the bert model and decide trainable and non trainable parameters
    '''

    def build(self, input_shape):
        self.bert = hub.KerasLayer(self.path_to_bert, trainable=self.trainable, name = "BertLayer") #Loading bert model
        variables = self.bert.variables
        trainable_variables = variables[-self.num_fine_tuning_layers :]     #setting parameters
        non_trainable_weights = variables[: -self.num_fine_tuning_layers]  
        
        for var in trainable_variables:
            self._trainable_weights.append(var)

        for var in non_trainable_weights:
            self._non_trainable_weights.append(var)

    
    def call(self, inputs, training = True):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_word_ids=input_ids,          #This tells you the token ids
            input_mask=input_mask,             #This tells the sentence area apart from the padding area
            input_type_ids=segment_ids,        #This tells the different segment information for multiple sentence input 
        )
        if self.pooling == "first":             #This returs the [CLS] token embeddings
            pooled = self.bert(inputs=bert_inputs)[
                "pooled_output"
            ]
        elif self.pooling == "mean":            #This returns embeddings of all the tokens
            result = self.bert(inputs=bert_inputs)[
                "sequence_output"
            ]

            pooled = result
        
        return pooled

def build_BERT(max_seq_length, n_tags, base_path, bert_path):

    input_ids = keras.layers.Input(shape = (max_seq_length), name = "input_ids")
    input_mask = keras.layers.Input(shape = (max_seq_length), name = "input_mask")
    segment_ids = keras.layers.Input(shape = (max_seq_length), name = "segment_ids")
    bert_input = [input_ids, input_mask, segment_ids]
    gs_folder_bert = base_path + "/" + bert_path
    bert_output = BertLayer(path_to_bert=gs_folder_bert)(bert_input)

    out = keras.layers.Dense(n_tags+1, activation='softmax')(bert_output)

    model = keras.models.Model(inputs = bert_input, outputs = out)

    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy)

    return model

def build_BERT_CRF(max_seq_length, n_tags, base_path, bert_path):

    input_ids = keras.layers.Input(shape = (max_seq_length), name = "input_ids")
    input_mask = keras.layers.Input(shape = (max_seq_length), name = "input_mask")
    segment_ids = keras.layers.Input(shape = (max_seq_length), name = "segment_ids")
    bert_input = [input_ids, input_mask, segment_ids]
    gs_folder_bert = base_path + "/" + bert_path
    bert_output = BertLayer(path_to_bert=gs_folder_bert)(bert_input)

    crf_layer = CRF(units=n_tags+1)
    out = crf_layer(bert_output)

    model = keras.models.Model(inputs = bert_input, outputs = out)

    model = crf.ModelWithCRFLoss(model)

    model.compile(optimizer="adam")

    return model