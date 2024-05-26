#!/usr/bin/env python3
"""Write the function that performs forward propagation for
bidirectional RNN"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for bidirectional RNN
    """

     t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize the hidden states container
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    # Initialize the hidden states
    Hf[0] = h_0
    Hb[-1] = h_t

    # forward direction
    for step in range(t):
        Hf[step + 1] = bi_cells.forward(Hf[step], X[step])

    # backward direction
    for step in range(t-1, -1, -1):
        Hb[step] = bi_cells.backward(Hb[step + 1], X[step])

    # concatenate hidden states
    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)

    # compute outputs
    Y = bi_cells.output(H)

    return H, Y
