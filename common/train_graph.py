# -*- coding: utf-8 -*-
# coded by simonjisu

import matplotlib.pylab as plt

def loss_graph(train_loss_list, train_acc_list=None, test_loss_list=None, test_acc_list=None):
    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=90)
    losses = [('b-', 'train_loss'), ('g-', 'test_loss')]
    accs = [('r-', 'train_acc'), ('y-', 'test_acc')]
    ax1.plot(train_loss_list, losses[0][0], label=losses[0][1])
    if test_loss_list:
        ax1.plot(test_loss_list, losses[1][0], label=losses[1][1])
    ax1.set_ylabel('Loss')

    if train_acc_list:
        ax2 = ax1.twinx()
        ax2.plot(train_acc_list, accs[0][0], label=accs[0][1])
        if test_acc_list:
            ax2.plot(train_acc_list, accs[1][0], label=accs[1][1])
        ax2.set_ylabel('Accuracy')

    plt.xlabel('Epoch')
    plt.title('LOSS & ACCURACY GRAPH' if train_acc_list else 'LOSS GRAPH', fontsize=15, y=1.01)
    plt.legend()
    plt.show()