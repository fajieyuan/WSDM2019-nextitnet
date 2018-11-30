# https://gaussic.github.io/2017/08/24/tensorflow-language-model/
from collections import Counter
import os
import numpy as np
import tensorflow as tf

class PTBModel(object):
    def __init__(self,config,num_steps,vocab_size,is_training=True):
        self.num_steps=num_steps
        self.vocab_size=vocab_size

        self.embedding_dim=config.embedding_dim
        self.hidden_dim=config.hidden_dim
        self.num_layers=config.num_layers
        self.rnn_model=config.rnn_model

        self.learning_rate=config.learning_rate


        self.placeholders()
        self.rnn()
        self.cost()
        self.optimize()
        # self.error()

    def placeholders(self):
        # self._inputs=tf.placeholder(tf.int32,[None,self.num_steps])
        # self._targets=tf.placeholder(tf.int32,[None,self.vocab_size])


        self.wholesession = tf.placeholder('int32',
                                           [None, None], name='wholesession')

        # a=self.t_sentence.get_shape()[1]*2


        source_sess = self.wholesession[:, 0:-1]
        target_sess = self.wholesession[:, -1:]

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        # output_keep_prob = config.keep_prob

        # source_embedding = tf.nn.embedding_lookup(self.wholesession,
        #                                           source_sess, name="source_embedding")
        # target_embedding=tf.nn.embedding_lookup(self.wholesession,
        #                                    target_sess, name="target_sess")


        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self._inputs = source_sess
        self._targets = target_sess

    def input_embedding(self):
        with tf.device("/cpu:0"):
            embedding=tf.get_variable("embedding",[self.vocab_size,self.embedding_dim],dtype=tf.float32)
            _inputs=tf.nn.embedding_lookup(embedding,self._inputs)
        return _inputs

    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTM(self.hidden_dim,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)
        def dropout_cell():
            if(self.rnn_model=='lstm'):
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout_keep_prob)
        cells=[dropout_cell() for _ in range(self.num_layers)]
        cell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
        _inputs=self.input_embedding()
        print _inputs.get_shape()
        _outputs,_=tf.nn.dynamic_rnn(cell,inputs=_inputs,dtype=tf.float32)

        # _outputs, _ = tf.contrib.rnn.static_rnn(cell,inputs=_inputs,dtype=tf.float32)
        last=_outputs[:,-1,:]
        logits=tf.layers.dense(inputs=last,units=self.vocab_size)
        prediction=tf.nn.softmax(logits)



        self._logits=logits
        self.probs_flat=prediction
        self.input_y = tf.reshape(self._targets, [-1])  # fajie addd

    def cost(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self.input_y)
        cost = tf.reduce_mean(cross_entropy)
        self.loss = cost

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,)
        self.optim = optimizer.minimize(self.loss)

    # def error(self):
    #     mistakes = tf.not_equal(tf.argmax(self._targets, 1), tf.argmax(self._pred, 1))
    #     self.erros = tf.reduce_mean(tf.cast(mistakes, tf.float32))
