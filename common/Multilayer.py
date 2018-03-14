# coding utf-8
from collections import OrderedDict, defaultdict
from common.layers import *
from common.utils_nn import *

class MLP(object):
    def __init__(self, input_size, hidden_size, output_size, activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0, use_dropout=False, dropout_ration=0.5, use_batchnorm=False, record=False):
        """
        input_size: int
        hidden_size: int (1 layer) may be a list (more than 2 layers)
        output_size: int
        activation: 
        weight_init_std: 
        weight_decay_lambda: 
        use_dropout: 
        dropout_ration: 
        use_batchnorm: 
        """
        self.input_size = input_size
        self.hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        self.hidden_layer_num = len(hidden_size)
        self.output_size = output_size
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.dropout_ration = dropout_ration
        self.params = {}

        # Weight Initialization
        self.__weight_init(weight_init_std)

        # Making Layer
        self.activation_layer = {'sigmoid': Sigmoid, 'relu': ReLu}
        self.__layer_init(activation)

        # activation record
        self.record = record
        self.activation_hists = defaultdict(list)
        self.backward_hists = defaultdict(list)


    def __weight_init(self, weight_init_std):
        """
        weight_init_std:
        - 'sigmoid' or 'xavier': Xavier weight initialization, sqrt(1/n)
        - 'relu' or 'he': He weight initialization, sqrt(2/n)
        * n: number of hidden layer node
        """
        all_size_list = [self.input_size] + self.hidden_size + [self.output_size]

        for i in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ['he', 'relu']:
                scale = np.sqrt(2.0 / all_size_list[i-1])
            elif str(weight_init_std).lower() in ['xavier', 'sigmoid']:
                scale = np.sqrt(1.0 / all_size_list[i-1])

            self.params['W' + str(i)] = scale * np.random.randn(all_size_list[i-1], all_size_list[i])
            self.params['b' + str(i)] = np.zeros(all_size_list[i])

    def __layer_init(self, activation):
        """
        layer structure: Affine + ( BatchNorm ) + Activation + ( Dropout )
        activation:
        - sigmoid
        - relu
        """
        self.layers = OrderedDict()
        for i in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])

            if self.use_batchnorm:
                self.params['gamma' + str(i)] = np.ones(self.hidden_size[i-1])
                self.params['beta' + str(i)] = np.zeros(self.hidden_size[i-1])
                self.layers['BatchNorm' + str(i)] = BatchNorm(self.params['gamma' + str(i)], self.params['beta' + str(i)])
                # self.layers['BatchNorm' + str(i)] = BatchNormalization(self.params['gamma' + str(i)],
                #                                                        self.params['beta' + str(i)])

            self.layers['Activation' + str(i)] = self.activation_layer[activation]()

            if self.use_dropout:
                self.layers['DropOut' + str(i)] = Dropout(self.dropout_ration)

        i = self.hidden_layer_num + 1
        self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])

        self.back_layers = OrderedDict(reversed(list(self.layers.items())))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
                if self.record & train_flg & ('Activation' in key):
                    self.activation_hists[key].append(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        for key, layer in self.back_layers.items():
            dout = layer.backward(dout)
            if self.record & ('Affine' in key):
                self.backward_hists[key].append(layer.dW)

        # save result
        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads['W' + str(i)] = self.layers['Affine' + str(i)].dW + self.weight_decay_lambda * self.params['W' + str(i)]
            grads['b' + str(i)] = self.layers['Affine' + str(i)].db

            if self.use_batchnorm and i != self.hidden_layer_num + 1:
                grads['gamma' + str(i)] = self.layers['BatchNorm' + str(i)].dgamma
                grads['beta' + str(i)] = self.layers['BatchNorm' + str(i)].dbeta

        return grads

    def numerical_gradient(self, X, T):
        """기울기를 구한다(수치 미분).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads