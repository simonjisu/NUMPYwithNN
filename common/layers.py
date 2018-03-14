# -*- coding: utf-8 -*-
# coded by simonjisu

import numpy as np
from common.utils_nn import softmax, cross_entropy_error, tanh


class ReLu(object):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx


class Affine(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(self.b, axis=0)

        return dx


class Linear(object):
    def __init__(self, W, b=None, bias=True):
        self.bias = bias
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        if self.bias:
            out = np.dot(self.x, self.W) + self.b
        else:
            out = np.dot(self.x, self.W)

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        if self.bias:
            self.db = np.sum(self.b, axis=0)
        else:
            self.db = None

        return dx



class Tanh(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1 - self.out**2)

        return dx


class SoftmaxWithLoss(object):
    def __init__(self):
        self.loss = None  # loss
        self.y = None  # softmax output
        self.t = None  # one-hot encoded answer vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class BatchNorm(object):
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        """

        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # for testing use learned mean and var
        self.running_mean = running_mean
        self.running_var = running_var

        # cache for backwardation
        self.batch_size = None
        self.xmu = None
        self.sq = None
        self.var = None
        self.std = None
        self.invstd = None
        self.xhat = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape

        out = self.__forward(x, train_flg)
        out = out.reshape(*self.input_shape)

        return out

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            self.running_mean = np.zeros(self.input_shape[1])
            self.running_var = np.zeros(self.input_shape[1])

        if train_flg:
            mu = x.mean(axis=0)
            xmu = x - mu
            sq = xmu**2
            var = np.mean(sq, axis=0)
            std = np.sqrt(var + 1e-6)
            invstd = 1.0 / std
            xhat = xmu * invstd

            self.batch_size = x.shape[0]
            self.xmu = xmu
            self.sq = sq
            self.var = var
            self.std = std
            self.invstd = invstd
            self.xhat = xhat
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # when test
            xmu = x - self.running_mean
            xhat = xmu / np.sqrt(self.running_var + 1e-6)

        out = self.gamma * xhat + self.beta
        return out

    def backward(self, dout):
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)

        return dx

    def __backward(self, dout):
        # step-9: out = scale + beta
        dbeta = dout.sum(axis=0)
        dscale = dout
        # step-8: scale = gamma * xhat
        dgamma = np.sum(self.xhat * dout, axis=0)
        dxhat = self.gamma * dscale
        # step-7: xhat = xmu * invstd
        dxmu1 = dxhat * self.invstd
        dinvstd = np.sum(dxhat * self.xmu, axis=0)
        # step-6: invstd = 1 / std
        dstd = dinvstd * (-self.invstd**2)
        # step-5: std = np.sqrt(var + 1e-6)
        dvar = -0.5 * dstd * (1 / np.sqrt(self.var + 1e-6))
        # step-4: var = sum(sq)
        dsq = (1.0 / self.batch_size) * np.ones(self.input_shape) * dvar
        # step-3: sq = xmu**2
        dxmu2 = dsq * 2 * self.xmu
        # step-2: xmu = x - mu
        dxmu = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu, axis=0)
        dx1 = dxmu * 1
        # step-1: mu = mean(x)
        dx2 = (1.0 / self.batch_size) * np.ones(self.input_shape) * dmu
        # step-0:
        dx = dx1 + dx2

        self.dbeta = dbeta
        self.dgamma = dgamma

        return dx

class BatchNormalization(object):
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # 합성곱 계층은 4차원, 완전연결 계층은 2차원

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Dropout(object):
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask