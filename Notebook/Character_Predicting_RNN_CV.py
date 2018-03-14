# -*- coding: utf-8 -*-
# coded by simonjisu

import os
import sys
import time
dir_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(dir_path)

import numpy as np
from common.SimpleRNN import Single_layer_RNN
from common.optimizer import Adam



x = 'hello world! nice to meet you! i love iron-man'

class chr_coding(object):
    def __init__(self):
        self._dict = None
        self._one_hot_matrix = None
        self._dict_reversed = None

    def fit(self, x):
        if isinstance(x, str):
            x = list(x)

        self._one_hot_matrix = np.eye(len(set(x)))
        self._dict = {d: i for i, d in enumerate(list(set(x)))}
        self._dict_reversed = {v: k for k, v in self._dict.items()}

    def encode(self, x):
        encoded_data = np.array([self._one_hot_matrix[self._dict[d]] for d in x])
        return encoded_data

    def decode(self, x, probs=None):
        if probs is None:
            decoded_data = self._dict_reversed[x]
        else:
            decoded_data = self._dict_reversed[np.argmax(probs)]
        return decoded_data

# accuracy calculation function
def get_accuracy(x, test_string):
    bool_ = np.array(list(x))[1:] == np.array(list(test_string))[1:]
    return bool_.sum() / len(bool_)

# train_function
def train(rnn, optim, option=True):
    total_loss_list = []
    total_acc_list = []
    for epoch in range(NUM_EPOCHS):
        test_string = 'h'
        # forward
        total_loss = rnn.loss(train_x, train_y)

        # backward
        if option:
            rnn.backward()
        else:
            rnn.backward_truncate()

        optim.update(rnn.params, rnn.params_summ)

        # test string
        predicted_idx = rnn.predict(train_x)
        for idx in predicted_idx:
            test_string += encoder.decode(idx)

        # get accuracy
        acc = get_accuracy(x, test_string)

        total_loss_list.append(total_loss)
        total_acc_list.append(acc)

    return total_acc_list[-1]

# encoding
encoder = chr_coding()
encoder.fit(x)
one_hot_data = encoder.encode(x)

# global parameters
NUM_CHECK = 5
NUM_EPOCHS = 3000
INPUT_SIZE = one_hot_data.shape[1]
HIDDEN_SIZE = 30
OUTPUT_SIZE = one_hot_data.shape[1]
SEQ_LEN = one_hot_data.shape[0] - 1
BPTT_TRUNCATE = 5

# data preparing
train_x = one_hot_data[:-1]
train_y = one_hot_data[1:]

## 3-models:
def get_models():
    # tanh + backward
    rnn1 = Single_layer_RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    optim1 = Adam()

    # tanh + backward_truncate
    rnn2 = Single_layer_RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                            bptt_truncate=BPTT_TRUNCATE)
    optim2 = Adam()

    # relu + backward_truncate
    rnn3 = Single_layer_RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                            bptt_truncate=BPTT_TRUNCATE, activation_func='relu')
    optim3 = Adam()

    labels = ['model1: tanh + backward', 'model2: tanh + backward_truncate', 'model3: relu + backward_truncate']
    rnns = [rnn1, rnn2, rnn3]
    optims = [optim1, optim2, optim3]
    return labels, rnns, optims

def main():
    print("Number of Checking:", NUM_CHECK)
    print("="*30)
    total_acc_array = []
    for i in range(NUM_CHECK):

        labels, rnns, optims = get_models()
        acc_list = []
        for j, (rnn, optim) in enumerate(zip(rnns, optims)):
            print("running #", str(i), "| " + labels[j])
            if j == 0:
                trun_key = True
            else:
                trun_key = False

            accuracy = train(rnn, optim, trun_key)
            acc_list.append(accuracy)

        total_acc_array.append(acc_list)

    total_acc_array = np.array(total_acc_array)

    for i in range(len(labels)):
        print("## " + labels[i] + " ##")
        print("Average Score:", np.mean(total_acc_array, axis=0)[i])
        print("Acc_list:", total_acc_array[:, i])


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))



