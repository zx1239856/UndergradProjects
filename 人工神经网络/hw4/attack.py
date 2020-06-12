import numpy as np
import tensorflow as tf
from common import config

mean, std = config.mean, config.std


class Attack():

    def __init__(self, model, batchsize, alpha, beta, gamma, CW_kappa, use_cross_entropy_loss, **kwargs):
        self.batchsize = batchsize
        self.model = model  # pretrained vgg model used as classifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.CW_kappa = CW_kappa
        self.use_cross_entropy_loss = use_cross_entropy_loss

    '''Build computation graph for generating adversarial examples'''

    def generate_graph(self, pre_noise, x, gt, target=None, **kwargs):
        noise = 10 * tf.tanh(pre_noise)
        x_noise = x + noise  ## add perturbation and get adversarial examples
        x_clip = tf.clip_by_value(x_noise, 0, 255)
        x_round = x_clip + tf.stop_gradient(
            x_clip // 1 - x_clip)  ##skip computing gradient wrt to rounded results(x_round) and only calculate the gradient wrt to x_clip
        x_norm = (x_round - mean) / (std + 1e-7)  ## normalize the image input for the classfier
        logits = self.model.build(x_norm)

        if self.use_cross_entropy_loss:
            loss = self.cross_entropy_loss(gt, target, logits, target is not None)
        else:
            loss = self.CW_attack_loss(gt, target, logits, target is not None)
        pertubations = (x_round - x) / 255
        pertubations_abs = tf.abs(x_round - x)  # for visualization
        reg = self.regularization(pertubations)
        loss = tf.reduce_mean(self.alpha * loss + reg)

        preds = tf.nn.softmax(logits)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32), gt), tf.float32))
        return acc, loss, x_round, pertubations_abs

    def regularization(self, pertubations):
        return self.beta * tf.reduce_sum(tf.abs(pertubations), axis=[1, 2, 3]) + \
               self.gamma * tf.reduce_sum(tf.square(pertubations), axis=[1, 2, 3])

    def cross_entropy_loss(self, ground_truth, target, logits, is_targeted_attack):
        # loss definition for both untargeted attack and targeted attack
        if is_targeted_attack:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
        else:
            loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth, logits=logits)
        return loss

    def CW_attack_loss(self, ground_truth, target, logits, is_targeted_attack):
        # loss definition for both untargeted attack and targeted attack
        gt = target if is_targeted_attack else ground_truth
        y_true_mask = tf.one_hot(gt, config.nr_class)
        y_other_mask = tf.one_hot(gt, config.nr_class, on_value=float('inf'))
        y_true = tf.reduce_sum(tf.multiply(y_true_mask, logits), 1)
        y_other = tf.reduce_max(logits - y_other_mask, 1)
        if is_targeted_attack:
            # make others smaller than target
            loss = tf.maximum(tf.cast(-self.CW_kappa, dtype=tf.float32), y_other - y_true)
        else:
            # make y_true smaller than others
            loss = tf.maximum(tf.cast(-self.CW_kappa, dtype=tf.float32), y_true - y_other)
        return loss

    '''Build a graph for evaluating the classification result of adversarial examples'''

    def evaluate(self, x, gt, **kwargs):
        x = (x - mean) / (std + 1e-7)
        logits = self.model.build(x)
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                                              tf.cast(tf.argmax(gt_one_hot, 1), dtype=tf.int32)), tf.float32))
        return acc
