# -*- coding: utf-8 -*-
import numpy as np

class SGD(object):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(object):
    def __init__(self, lr=0.01, gamma=0.9):
        """gamma: momentum term, how much momentum you want to give, default is 0.9"""
        self.lr = lr
        self.gamma = gamma
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.gamma * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class RMSProp(object):
    def __init__(self, lr=0.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma  # decay term
        self.G = None
        self.epsilon = 1e-6  # 0으로 나누눈 것을 방지

    def update(self, params, grads):
        if self.G is None:
            self.G = {}
            for key, val in params.items():
                self.G[key] = np.zeros_like(val)

        for key in params.keys():
            self.G[key] += self.gamma * self.G[key] + (1 - self.gamma) * (grads[key] * grads[key])
            params[key] -= self.lr * grads[key] / np.sqrt(self.G[key] + self.epsilon)


class AdaDelta(object):
    def __init__(self, gamma=0.9):
        """
        https://arxiv.org/pdf/1212.5701.pdf
        """
        self.gamma = gamma  # decay term
        self.G = None  # accumulated gradients
        self.s = None  # accumulated updates
        self.del_W = None
        self.epsilon = 1e-6  # 0으로 나누눈 것을 방지
        self.iter = 0

    def update(self, params, grads):
        if (self.G is None) | (self.s is None) | (self.del_W is None):
            # Initialize accumulation variables
            self.G = {}
            self.s = {}
            self.del_W = {}
            for key, val in params.items():
                self.G[key] = np.zeros_like(val)
                self.s[key] = np.zeros_like(val)
                self.del_W[key] = np.zeros_like(val)

        for key in params.keys():
            self.G[key] += self.gamma * self.G[key] + (1 - self.gamma) * (grads[key] * grads[key])
            self.del_W[key] = -(np.sqrt(self.s[key] + self.epsilon) / np.sqrt(self.G[key] + self.epsilon)) * grads[key]
            self.s[key] += self.gamma * self.s[key] + (1 - self.gamma) * self.del_W[key] ** 2
            params[key] += self.del_W[key]


class Adagrad(object):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        self.epsilon = 1e-6  # 0으로 나누눈 것을 방지

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + self.epsilon)


class Adam(object):
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)