#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np

from custom_vgg16_bn import Model
from dataset import Dataset
from common import config
from attack import Attack


# import cv2


def get_dataset_batch(ds_name):
    dataset = Dataset(ds_name)
    ds_gnr = dataset.load().instance_generator
    ds = tf.data.Dataset.from_generator(ds_gnr, output_types=(tf.float32, tf.int32))
    ds = ds.repeat(1)
    ds = ds.batch(config.minibatch_size)
    ds_iter = ds.make_one_shot_iterator()
    sample_gnr = ds_iter.get_next()
    return sample_gnr, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr_epoch", type=int, default=500,
                        help="you may need to increase nr_epoch to 4000 or more for targeted adversarial attacks")
    parser.add_argument("--alpha", type=float, default=0,
                        help="coefficient of either cross entropy loss or C&W attack loss")
    parser.add_argument("--beta", type=float, default=0, help="coefficient of lasso regularization")
    parser.add_argument("--gamma", type=float, default=0, help="coefficient of ridge regularization")
    parser.add_argument("--CW_kappa", type=float, default=0, help="hyperparameter for C&W attack loss")
    parser.add_argument("--use_cross_entropy_loss", action='store_true')
    parser.add_argument("--targeted_attack", action='store_true')
    args = parser.parse_args()

    ## load dataset
    train_batch_gnr, train_set = get_dataset_batch(ds_name='train')

    data = tf.placeholder(tf.float32, shape=(None,) + config.image_shape + (config.nr_channel,), name='data')
    label = tf.placeholder(tf.int32, shape=(None,), name='label')  # placeholder for targeted label
    gt = tf.placeholder(tf.int32, shape=(None,), name='gt')

    pre_noise = tf.Variable(
        tf.zeros((config.minibatch_size, config.image_shape[0], config.image_shape[1], config.nr_channel),
                 dtype=tf.float32))
    model = Model()
    attack = Attack(model, config.minibatch_size, args.alpha, args.beta, args.gamma, args.CW_kappa,
                    args.use_cross_entropy_loss)
    target = label if args.targeted_attack else None
    acc, loss, adv, noise = attack.generate_graph(pre_noise, data, gt, target)
    acc_gt = attack.evaluate(data, gt)

    placeholders = {
        'data': data,
        'label': label,
        'gt': gt,
    }

    lr = 1e-2
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss, [pre_noise])
    train = opt.apply_gradients(grads)
    ## init tensorboard
    tf.summary.scalar('loss', loss)
    tf.summary.image('input_image', data, family='attack')
    tf.summary.image('attack_image', adv, family='attack')
    tf.summary.image('noise', noise, family='attack')
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'run_{}'.format(time.time()), 'train'),
                                         tf.get_default_graph())

    ## create a session
    tf.set_random_seed(12345)  # ensure consistent results
    succ = 0
    noise_l1 = 0
    noise_l2 = 0
    noise_l_inf = 0
    tot = 0
    with tf.Session() as sess:

        assert train_set.minibatch_size == 1
        for idx in range(train_set.minibatches):
            global_cnt = 0

            sess.run(tf.global_variables_initializer())  # init all variables
            images, gt = sess.run(train_batch_gnr)
            tf.print(gt, output_stream=sys.stderr)
            if acc_gt.eval(feed_dict={placeholders['data']: images, placeholders['gt']: gt}) < 0.5:
                continue
            else:
                tot += 1

            if args.targeted_attack:
                labels = (gt + 5) % config.nr_class
            else:
                labels = gt

            min_distortion = np.inf
            min_l1 = np.inf
            min_l2 = np.inf
            min_linf = np.inf

            for epoch in range(1, args.nr_epoch + 1):
                global_cnt += 1
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    placeholders['gt']: gt,
                }

                _, accuracy, loss_batch, adv_examples, summary = sess.run([train, acc, loss, adv, merged],
                                                                          feed_dict=feed_dict)

                if global_cnt % config.show_interval == 0:
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{}/{}, {}".format(idx, train_set.minibatches, epoch),
                        'loss: {:.3f}'.format(loss_batch),
                        'acc: {:3f}'.format(accuracy),
                    )

                successful = acc_gt.eval(feed_dict={placeholders['data']: adv_examples, placeholders[
                    'gt']: labels}) > 0.5 if args.targeted_attack else acc_gt.eval(
                    feed_dict={placeholders['data']: adv_examples, placeholders['gt']: gt}) < 0.5

                if successful:
                    l1 = np.sum(np.abs((adv_examples - images) / 255))
                    l2_square = np.sum(((adv_examples - images) / 255) ** 2)
                    distortion = args.beta * l1 + args.gamma * l2_square
                    if distortion < min_distortion:
                        min_distortion = distortion
                        min_l1 = min(min_l1, l1)
                        min_l2 = min(min_l2, np.sqrt(l2_square))
                        min_linf = min(min_linf, np.max((adv_examples - images) / 255))

            print('Training for batch {} is done'.format(idx))
            sys.stdout.flush()

            if min_distortion != np.inf:
                succ += 1
                noise_l1 += min_l1
                noise_l2 += min_l2
                noise_l_inf += min_linf
        for attr in dir(args):
            if not attr.startswith("_"):
                print("{}: {}".format(attr, getattr(args, attr)))
        print('Success rate: {}'.format(succ / tot))
        print('Noise l1-norm: {}'.format(noise_l1 / tot))
        print('Noise l2-norm: {}'.format(noise_l2 / tot))
        print('Noise l-inf: {}'.format(noise_l_inf / tot))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
