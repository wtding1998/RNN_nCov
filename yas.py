#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd

from module import MLP, MLP_tanh, MLP_sigmoid


class yas(torch.nn.Module):
    def __init__(self, nd, nz, nhid, nlayers, activation="relu", dropout=0.5):
        super(yas, self).__init__()
        if activation == "relu":
            self.encoder = MLP(nd, nhid, nz, nlayers, dropout=dropout)
            self.decoder = MLP(nz, nhid, nd, nlayers, dropout=dropout)
        elif activation == "tanh":
            self.encoder = MLP_tanh(nd, nhid, nz, nlayers, dropout=dropout)
            self.decoder = MLP_tanh(nz, nhid, nd, nlayers, dropout=dropout)
        elif activation == "sigmoid":
            self.encoder = MLP_sigmoid(nd, nhid, nz, nlayers, dropout=dropout)
            self.decoder = MLP_sigmoid(nz, nhid, nd, nlayers, dropout=dropout)
            np.stack()
