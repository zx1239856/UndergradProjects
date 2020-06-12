# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS
INPUT_DIM = 32 * 32 * 3
OUTPUT_DIM = 10


class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, INPUT_DIM])
        self.y_ = tf.placeholder(tf.int32, [None])

        self.loss, self.pred, self.acc = self.forward(is_train=True)  # train
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False, reuse=True)  # test

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(self.params, print_info=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                                var_list=self.params)  # Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
        with tf.variable_scope("model", reuse=reuse):
            # Your Linear Layer
            res = tf.layers.dense(self.x_, 512,
                                  kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
            res = batch_normalization_layer(res, is_train)
            # Your ReLU Layer
            res = tf.nn.relu(res)
            # Your Dropout Layer: use dropout_layer function
            res = dropout_layer(res, FLAGS.drop_rate, is_train)
            # Your Linear Layer
            logits = tf.layers.dense(res, OUTPUT_DIM,
                                     kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        return loss, pred, acc


def batch_normalization_layer(incoming, is_train=True):
    # implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should use mu and sigma calculated based on mini-batch
    #       If isTrain is False, you must use mu and sigma estimated from training data
    return tf.layers.batch_normalization(incoming, 1, training=is_train)


def dropout_layer(incoming, drop_rate, is_train=True):
    # implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    return tf.layers.dropout(incoming, drop_rate, training=is_train)
