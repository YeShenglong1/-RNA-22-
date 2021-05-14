import math, random
from const import START, STOP

import numpy as np
from collections import defaultdict, OrderedDict
from pprint import pprint

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
#from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

class Network(Layer):
    def __init__(self,
                 sequence_vocabulary, bracket_vocabulary,
                 dmodel=128,
                 layers=8,
                 dropout=0.15,
                 ):
        super(Network, self).__init__()
        self.sequence_vocabulary = sequence_vocabulary
        self.bracket_vocabulary = bracket_vocabulary
        self.dropout_rate = dropout
        self.model_size = dmodel
        self.layers = layers
        
    def forward(self, seq, dot):
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True)

        emb_seq = paddle.fluid.layers.fc(emb_seq, size=self.model_size*4)
        emb_seq_fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb_seq, size=self.model_size*4, use_peepholes=True, is_reverse=False)
        emb_seq_back, cell = paddle.fluid.layers.dynamic_lstm(input=emb_seq, size=self.model_size*4, use_peepholes=True, is_reverse=True)
        emb_seq_fwd = paddle.fluid.layers.fc(emb_seq_fwd, size=self.model_size*4)
        emb_seq_back = paddle.fluid.layers.fc(emb_seq_back, size=self.model_size*4)
        emb_seq_fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb_seq_fwd, size=self.model_size*4, use_peepholes=True, is_reverse=False)
        emb_seq_back, cell = paddle.fluid.layers.dynamic_lstm(input=emb_seq_back, size=self.model_size*4, use_peepholes=True, is_reverse=True)
        emb_seq = paddle.fluid.layers.concat(input=[emb_seq_fwd, emb_seq_back], axis=1)
        emb_seq = paddle.fluid.layers.fc(emb_seq, size=self.model_size, act="relu")

        emb_dot = paddle.fluid.layers.fc(emb_dot, size=self.model_size*4)
        emb_dot_fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb_dot, size=self.model_size*4, use_peepholes=True, is_reverse=False)
        emb_dot_back, cell = paddle.fluid.layers.dynamic_lstm(input=emb_dot, size=self.model_size*4, use_peepholes=True, is_reverse=True)
        emb_dot_fwd = paddle.fluid.layers.fc(emb_dot_fwd, size=self.model_size*4)
        emb_dot_back = paddle.fluid.layers.fc(emb_dot_back, size=self.model_size*4)
        emb_dot_fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb_dot_fwd, size=self.model_size*4, use_peepholes=True, is_reverse=False)
        emb_dot_back, cell = paddle.fluid.layers.dynamic_lstm(input=emb_dot_back, size=self.model_size*4, use_peepholes=True, is_reverse=True)
        emb_dot = paddle.fluid.layers.concat(input=[emb_dot_fwd, emb_dot_back], axis=1)
        emb_dot = paddle.fluid.layers.fc(emb_dot, size=self.model_size, act="relu")

        emb = paddle.fluid.layers.concat(input=[emb_seq,emb_dot], axis=1)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        for _ in range(self.layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size*4)
            fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size*4, use_peepholes=True, is_reverse=False)
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size*4, use_peepholes=True, is_reverse=True)
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)
            emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        ff_out = paddle.fluid.layers.fc(emb, size=2, act="relu")
        soft_out = paddle.fluid.layers.softmax(ff_out, axis=1)
        return soft_out[:,0]

