import tensorflow as tf
import numpy as np

'''
including verical and horizonal cnn
'''
class TextCNN_hv(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, loss_type, l2_reg_lambda):

        # Placeholders for input, output and dropout



        self.wholesession = tf.placeholder('int32',
                                         [None, None], name='wholesession')

        # a=self.t_sentence.get_shape()[1]*2


        source_sess = self.wholesession[:, 0:-1]
        target_sess = self.wholesession[:, -1:]

        new_sequence_length=sequence_length-1

        # source_embedding = tf.nn.embedding_lookup(self.wholesession,
        #                                           source_sess, name="source_embedding")
        # target_embedding=tf.nn.embedding_lookup(self.wholesession,
        #                                    target_sess, name="target_sess")


        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_x=source_sess
        self.input_y=target_sess

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.loss_type = loss_type
        self.l2_reg_lambda = l2_reg_lambda


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # http://www.infoq.com/cn/articles/introduction-of-tensorflow-part4   how to use cnn
                # new shape after conv2d[?, new_sequence_length - filter_size + 1, 1, 1]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # new shape after max_pool[?, 1, 1, num_filters]
                # be carefyul, the  new_sequence_length has changed because of wholesession[:, 0:-1]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, new_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) #shape=[batch_size, 384]
        # design the veritcal cnn
        with tf.name_scope("conv-verical" ):
            filter_shape = [new_sequence_length, 1, 1, 1]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        self.vcnn_flat= tf.reshape(h, [-1, embedding_size])
        self.final=tf.concat([self.h_pool_flat,self.vcnn_flat],1) #shape=[batch_size, 384+100]




        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.final, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total+embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.input_y = tf.reshape(self.input_y, [-1])
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(self.loss + l2_reg_lambda * l2_loss)

            self.probs_flat = tf.nn.softmax(self.scores)
            self.arg_max_prediction = tf.argmax(self.probs_flat, 1)


        # Calculate mean cross-entropy loss
        # with tf.name_scope("loss"):
        #     if self.loss_type == 'square_loss':
        #         if self.l2_reg_lambda > 0:
        #             self.loss = tf.nn.l2_loss(
        #                 tf.subtract(self.input_y, self.scores)) +  l2_reg_lambda * l2_loss  # regulizer
        #         else:
        #             self.loss = tf.nn.l2_loss(tf.subtract(self.input_y, self.scores))






        # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
