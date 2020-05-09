from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop, adam
import numpy as np

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
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))
    return model

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
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))
    return model

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
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))
    return model

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
    model.add(Activation(activation))
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))
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

