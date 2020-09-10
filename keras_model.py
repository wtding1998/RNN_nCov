from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

class LSTM_module_cl():
    def __init__(self, ninput, nhid, nlayers, nout, test_input, nx, nd, activation='relu', lr=1e-3, dropout=0.5):
        assert (nlayers > 0)
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nout
        self.activation = activation
        self.lr = lr
        self.dropout = dropout
        self.test_input = test_input
        self.nx = nx
        self.nd = nd
        self.network = self.init_network()

    def init_network(self):
        network=Sequential()
        if self.nlayers == 1:
            network.add(LSTM(
                self.nhid,
                input_shape=(None, self.ninput),
                return_sequences=False))
        else:
            network.add(LSTM(
                self.nhid,
                input_shape=(None, self.ninput),
                return_sequences=True))
            for i in range(self.nlayers - 2):
                network.add(LSTM(
                    self.nhid,
                    return_sequences=True))
            network.add(LSTM(
                self.nhid,
                return_sequences=False))
        network.add(Dense(self.nout))
        network.add(Activation(self.activation))
        network.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return network

    def generate(self, nsteps):
        pred = []
        last_sequence = self.test_input[np.newaxis, ...]
        for i in range(nsteps):
            new_pred = self.network.predict(last_sequence)
            pred.append(new_pred)
            new_pred = new_pred[np.newaxis, ...]
            last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
        pred = np.concatenate(pred, axis=0)
        pred = np.reshape(pred, (nsteps, self.nx, self.nd))
        return pred

def LSTM_module(ninput, nhid, nlayers, nout, activation='relu', lr=1e-3, dropout=0.5):
    assert (nlayers > 0)
    model=Sequential()
    if nlayers == 1:
        model.add(LSTM(
            nhid,
            input_shape=(None, ninput),
            return_sequences=False))
    else:
        model.add(LSTM(
            nhid,
            input_shape=(None, ninput),
            return_sequences=True))
        for i in range(nlayers - 2):
            model.add(LSTM(
                nhid,
                return_sequences=True))
        model.add(LSTM(
            nhid,
            return_sequences=False))
    model.add(Dense(nout))
    model.add(Activation(activation))
    model.compile(loss="mse", optimizer=Adam(lr=lr))
    return model

class GRU_module_cl():
    def __init__(self, ninput, nhid, nlayers, nout, test_input, nx, nd, activation='relu', lr=1e-3, dropout=0.5):
        assert (nlayers > 0)
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nout
        self.activation = activation
        self.lr = lr
        self.dropout = dropout
        self.test_input = test_input
        self.nx = nx
        self.nd = nd
        self.network = self.init_network()

    def init_network(self):
        network=Sequential()
        if self.nlayers == 1:
            network.add(GRU(
                self.nhid,
                input_shape=(None, self.ninput),
                return_sequences=False))
        else:
            network.add(GRU(
                self.nhid,
                input_shape=(None, self.ninput),
                return_sequences=True))
            for i in range(self.nlayers - 2):
                network.add(GRU(
                    self.nhid,
                    return_sequences=True))
            network.add(GRU(
                self.nhid,
                return_sequences=False))
        network.add(Dense(self.nout))
        network.add(Activation(self.activation))
        network.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return network

    def generate(self, nsteps):
        pred = []
        last_sequence = self.test_input[np.newaxis, ...]
        for i in range(nsteps):
            new_pred = self.network.predict(last_sequence)
            pred.append(new_pred)
            new_pred = new_pred[np.newaxis, ...]
            last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
        pred = np.concatenate(pred, axis=0)
        pred = np.reshape(pred, (nsteps, self.nx, self.nd))
        return pred



def GRU_module(ninput, nhid, nlayers, nout, activation='relu', lr=1e-3, dropout=0.5):
    assert (nlayers > 0)
    model=Sequential()
    if nlayers == 1:
        model.add(GRU(
            nhid,
            input_shape=(None, ninput),
            return_sequences=False))
    else:
        model.add(GRU(
            nhid,
            input_shape=(None, ninput),
            return_sequences=True))
        for i in range(nlayers - 2):
            model.add(GRU(
                nhid,
                return_sequences=True))
        model.add(GRU(
            nhid,
            return_sequences=False))
    model.add(Dense(nout))
    model.add(Activation(activation))
    model.compile(loss="mse", optimizer=Adam(lr=lr))
    return model

class LSTM_Linear_cl():
    def __init__(self, ninput, nhid, nlayers, nout, test_input, nx, nd, activation='relu', lr=1e-3, dropout=0.5):
        assert (nlayers > 0)
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nout
        self.activation = activation
        self.lr = lr
        self.dropout = dropout
        self.test_input = test_input
        self.nx = nx
        self.nd = nd
        self.network = self.init_network()

    def init_network(self):
        network=Sequential()
        if nlayers == 1:
            network.add(LSTM(
                nhid,
                input_shape=(None, ninput),
                return_sequences=False))
        else:
            network.add(LSTM(
                nhid,
                input_shape=(None, ninput),
                return_sequences=True))
            network.add(Dense(nhid))
            network.add(Activation(activation))
            network.add(Dropout(dropout))
            for i in range(nlayers - 2):
                network.add(LSTM(
                    nhid,
                    return_sequences=True))
                network.add(Dense(nhid))
                network.add(Activation(activation))
                network.add(Dropout(dropout))

            network.add(LSTM(
                nhid,
                return_sequences=False))
        network.add(Dense(nout))
        network.add(Activation(activation))
        network.compile(loss="mse", optimizer=Adam(lr=lr))
        return network

    def generate(self, nsteps):
        pred = []
        last_sequence = self.test_input[np.newaxis, ...]
        for i in range(nsteps):
            new_pred = self.network.predict(last_sequence)
            pred.append(new_pred)
            new_pred = new_pred[np.newaxis, ...]
            last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
        pred = np.concatenate(pred, axis=0)
        pred = np.reshape(pred, (nsteps, self.nx, self.nd))
        return pred

def LSTM_Linear(ninput, nhid, nlayers, nout, activation='relu', lr=1e-3, dropout=0.5):
    assert (nlayers > 0)
    model=Sequential()
    if nlayers == 1:
        model.add(LSTM(
            nhid,
            input_shape=(None, ninput),
            return_sequences=False))
    else:
        model.add(LSTM(
            nhid,
            input_shape=(None, ninput),
            return_sequences=True))
        model.add(Dense(nhid))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        for i in range(nlayers - 2):
            model.add(LSTM(
                nhid,
                return_sequences=True))
            model.add(Dense(nhid))
            model.add(Activation(activation))
            model.add(Dropout(dropout))

        model.add(LSTM(
            nhid,
            return_sequences=False))
    model.add(Dense(nout))
    model.add(Activation(activation))
    model.compile(loss="mse", optimizer=Adam(lr=lr))
    return model

class GRU_Linear_cl():
    def __init__(self, ninput, nhid, nlayers, nout, test_input, nx, nd, activation='relu', lr=1e-3, dropout=0.5):
        assert (nlayers > 0)
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nout
        self.activation = activation
        self.lr = lr
        self.dropout = dropout
        self.test_input = test_input
        self.nx = nx
        self.nd = nd
        self.network = self.init_network()

    def init_network(self):
        network=Sequential()
        if nlayers == 1:
            network.add(GRU(
                nhid,
                input_shape=(None, ninput),
                return_sequences=False))
        else:
            network.add(GRU(
                nhid,
                input_shape=(None, ninput),
                return_sequences=True))
            network.add(Dense(nhid))
            network.add(Activation(activation))
            network.add(Dropout(dropout))
            for i in range(nlayers - 2):
                network.add(GRU(
                    nhid,
                    return_sequences=True))
                network.add(Dense(nhid))
                network.add(Activation(activation))
                network.add(Dropout(dropout))

            network.add(GRU(
                nhid,
                return_sequences=False))
        network.add(Dense(nout))
        network.add(Activation(activation))
        network.compile(loss="mse", optimizer=Adam(lr=lr))
        return network

    def generate(self, nsteps):
        pred = []
        last_sequence = self.test_input[np.newaxis, ...]
        for i in range(nsteps):
            new_pred = self.network.predict(last_sequence)
            pred.append(new_pred)
            new_pred = new_pred[np.newaxis, ...]
            last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
        pred = np.concatenate(pred, axis=0)
        pred = np.reshape(pred, (nsteps, self.nx, self.nd))
        return pred

def GRU_Linear(ninput, nhid, nlayers, nout, activation='relu', lr=1e-3, dropout=0.5):
    assert (nlayers > 0)
    model=Sequential()
    if nlayers == 1:
        model.add(GRU(
            nhid,
            input_shape=(None, ninput),
            return_sequences=False))
    else:
        model.add(GRU(
            nhid,
            input_shape=(None, ninput),
            return_sequences=True))
        model.add(Dense(nhid))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        for i in range(nlayers - 2):
            model.add(GRU(
                nhid,
                return_sequences=True))
            model.add(Dense(nhid))
            model.add(Activation(activation))
            model.add(Dropout(dropout))

        model.add(GRU(
            nhid,
            return_sequences=False))
    model.add(Dense(nout))
    model.compile(loss="mse", optimizer=Adam(lr=lr))
    return model

if __name__ == "__main__":
    # --- test ---
    model = LSTM_Linear(10, 10, 2, 1)
    model.summary()
    data = np.random.rand(10, 3, 10)
    out = model.predict(data)
    print(out.shape)

# # 1st layer
# if opt.rnn_model == 'LSTM':
#         model.add(LSTM(
#             opt.nhid,
#             input_shape=(None, opt.nx*opt.nd),
#             return_sequences=True))
# elif opt.rnn_model == 'GRU':
#         model.add(GRU(
#             opt.nhid,
#             input_shape=(None, opt.nx*opt.nd),
#             return_sequences=True))
# model.add(Dense(opt.nx*opt.nd, activation=opt.activation))
# model.add(Dropout(opt.dropout))
# # middle layers
# for i in range(opt.nlayers-2):
#     if opt.rnn_model == 'LSTM':
#         model.add(LSTM(
#             opt.nhid,
#             return_sequences=True))
#     elif opt.rnn_model == 'GRU':
#         model.add(GRU(
#             opt.nhid,
#             return_sequences=True))
#     model.add(Dense(opt.nx*opt.nd, activation=opt.activation))
#     model.add(Dropout(opt.dropout))

# # final layer
# if opt.rnn_model == 'LSTM':
#         model.add(LSTM(
#             opt.nhid,
#             return_sequences=False))
# elif opt.rnn_model == 'GRU':
#         model.add(GRU(
#             opt.nhid,
#             return_sequences=False))
# model.add(Dropout(opt.dropout))
