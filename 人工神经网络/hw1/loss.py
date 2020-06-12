from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Compute loss w.r.t. input and target'''
        norm = np.linalg.norm(target - input, axis=1)
        return np.sum(norm * norm) / (2.0 * input.shape[0])

    def backward(self, input, target):
        '''Compute derivatives w.r.t. input'''
        return (input - target) / input.shape[0]


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Compute loss w.r.t. input and target'''
        exp = np.exp(input)
        h = exp / np.sum(exp, axis=1, keepdims=True) # probabilities via softmax
        self._saved_h = h ## save h for backward
        return np.sum(-target * np.log(h)) / input.shape[0]

    def backward(self, input, target):
        '''Compute derivatives w.r.t. input'''
        return (self._saved_h / np.sum(target, axis=1, keepdims=True) - target) / input.shape[0]
