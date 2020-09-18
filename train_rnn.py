import datetime
import json
import os
import random
from collections import OrderedDict, defaultdict

import configargparse
import numpy as np
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import trange
# from tensorflow import set_random_seed
# import tensorflow as tf

from get_dataset import get_keras_dataset, get_true
from keras_model import *
from utils import (DotDict, Logger_keras, boolean_string, get_dir, get_time,
                   rmse_np, rmse_np_like_torch, shuffle_list, time_dir)


#######################################################################################################################
# Options - CUDA - Random seed
#######################################################################################################################
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='mar_rnn')
p.add('--normalize', type=str, help='normaize method : vairance | min_max', default='variance')
p.add('--nt_train', type=int, help='time for training', default=50)
p.add('--start_time', type=int, help='time for training', default=0)
p.add('--increase', type=boolean_string, help='whether to use daily increase data', default=False)
p.add('--reduce', type=boolean_string, help='whether to use every province data', default=True)

# -- xp
p.add('--outputdir', type=str, help='path to save xp', default='default')
p.add('--xp', type=str, help='xp name', default='rnn')
# p.add('--dir_auto', type=boolean_string, help='dataset_model', default=True)
p.add('--xp_auto', type=boolean_string, help='time', default=False)
p.add('--xp_time', type=boolean_string, help='xp_time', default=True)
p.add('--auto', type=boolean_string, help='dataset_model + time', default=False)
# -- model
p.add('--seq_length', type=int, help='sequence length', default=5)
p.add('--nhid', type=int, help='dynamic function hidden size', default=50)
p.add('--nlayers', type=int, help='dynamic function num layers', default=2)
p.add('--dropout', type=float, help='dropout rate', default=0.5)
p.add('--activation', type=str, help='activation function : relu | tanh | sigmoid', default='tanh')
p.add('--rnn_model', type=str, help='choose rnn model : LSTM(GRU)_module | LSTM(GRU)_Linear ', default='GRU_module')
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-3)
p.add('--validation_ratio', type=float, help='validation rate', default=0.1)
p.add('--clip_value', type=float, help='clip_value for learning', default=5.0)
p.add('--patience', type=int, help='patience in early stopping', default=150)

# -- learning
p.add('--batch_size', type=int, default=10, help='batch size')
p.add('--nepoch', type=int, default=1, help='number of epochs to train for')
# -- seed
p.add('--manualSeed', type=int, help='manual seed')
# -- logs
p.add('--checkpoint_interval', type=int, default=700, help='check point interval')
p.add('--log', type=boolean_string, default=True, help='log')
# parse
opt = DotDict(vars(p.parse_args()))


# seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
# set_random_seed(opt.manualSeed)
#######################################################################################################################
# Data
#######################################################################################################################
# -- load data

if opt.increase:
    opt.dataset = opt.dataset + '_increase'

setup, (train_input, train_output), (val_input, val_output), (test_input, test_data)= get_keras_dataset(opt.datadir, opt.dataset, opt.nt_train, opt.seq_length, start_time=opt.start_time, normalize=opt.normalize, reduce=opt.reduce)

for k, v in setup.items():
    opt[k] = v

if opt.outputdir == 'default':
    opt.outputdir = opt.dataset + "_" + opt.rnn_model
opt.outputdir = get_dir(opt.outputdir)

if opt.xp_time:
    opt.xp = opt.xp + "_" + get_time()
if opt.xp_auto:
    opt.xp = get_time()
if opt.auto_all:
    opt.outputdir = opt.dataset + "_" + opt.rnn_model 
    opt.xp = get_time()
opt.xp = 'keras-' + opt.xp
opt.start = time_dir()
start_st = datetime.datetime.now()
opt.st = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
#######################################################################################################################
# Model
#######################################################################################################################
if opt.rnn_model == 'GRU_module':
    model = GRU_module(opt.nx*opt.nd, opt.nhid, opt.nlayers, opt.nx*opt.nd, activation=opt.activation, lr=opt.lr, dropout=opt.dropout)

if opt.rnn_model == 'LSTM_module':
    # model = LSTM_module(opt.nx*opt.nd, opt.nhid, opt.nlayers, opt.nx*opt.nd, activation=opt.activation, lr=opt.lr, dropout=opt.dropout)
    model = LSTM_module(opt.nx*opt.nd, opt.nhid, opt.nlayers, opt.nx*opt.nd, activation=opt.activation, lr=opt.lr, dropout=opt.dropout)
    
if opt.rnn_model == 'GRU_Linear':
    model = GRU_Linear(opt.nx*opt.nd, opt.nhid, opt.nlayers, opt.nx*opt.nd, activation=opt.activation, lr=opt.lr, dropout=opt.dropout)

if opt.rnn_model == 'LSTM_Linear':
    model = LSTM_Linear(opt.nx*opt.nd, opt.nhid, opt.nlayers, opt.nx*opt.nd, activation=opt.activation, lr=opt.lr, dropout=opt.dropout)

# model = Sequential()
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

# model.add(Dense(
#     opt.nx*opt.nd))
# model.add(Activation(opt.activation))
# model.compile(loss="mse", optimizer=RMSprop(lr=opt.lr))

#######################################################################################################################
# Logs
#######################################################################################################################
logger = Logger_keras(get_dir(opt.outputdir), opt.xp, opt.checkpoint_interval)
#######################################################################################################################
# Training
#######################################################################################################################

early_stopping = EarlyStopping(monitor='val_loss', patience=opt.patience, verbose=2)

model_history = model.fit(
    train_input, train_output, validation_data=(val_input, val_output),
    batch_size=opt.batch_size, epochs=opt.nepoch, callbacks=[early_stopping], verbose=2)

#######################################################################################################################
# Test
#######################################################################################################################
# generate pred
pred = []
last_sequence = test_input[np.newaxis, ...]
for i in range(opt.nt - opt.nt_train):
    new_pred = model.predict(last_sequence)
    pred.append(new_pred)
    new_pred = new_pred[np.newaxis, ...]
    last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
pred = np.concatenate(pred, axis=0)
pred = np.reshape(pred, (opt.nt - opt.nt_train, opt.nx, opt.nd))
test_data = np.reshape(test_data, (opt.nt - opt.nt_train, opt.nx, opt.nd))
# print(pred)
opt.rmse_score = rmse_np_like_torch(pred, test_data)
opt.sum_score = np.linalg.norm(pred - test_data) / pred.shape[0]
if opt.normalize == 'max_min':
    pred = pred * (opt.max - opt.min) + opt.mean
    opt.final_rmse_loss = opt.rmse_score * (opt.max - opt.min)
    opt.final_sum_loss = opt.sum_score * (opt.max - opt.min)

if opt.normalize == 'variance':
    pred = pred * opt.std + opt.mean
    opt.final_rmse_loss = opt.rmse_score * opt.std
    opt.final_sum_loss = opt.sum_score * opt.std

train_loss_history = model_history.history['loss']
test_loss_history = model_history.history['val_loss']
opt.minrmse = min(test_loss_history)
opt.min_rmse_epoch = test_loss_history.index(opt.minrmse)
opt.rmse_score = test_loss_history[-1]
if opt.log:
    logger.log('train_loss.epoch', train_loss_history)
    logger.log('test_loss.epoch', test_loss_history)
    
opt.end = time_dir()
end_st = datetime.datetime.now()
opt.et = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
opt.time = str(end_st - start_st)

with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)

logger.save(model)
if opt.increase:
    for i, time_data in enumerate(opt.datas_order):
        d_pred = pred[:,:, i]
        np.savetxt(os.path.join(get_dir(opt.outputdir), opt.xp, 'increase_' + opt.datas_order[i] +  '.txt'), d_pred, delimiter=',')
else:
    for i, time_data in enumerate(opt.datas_order):
        d_pred = pred[:,:, i]
        np.savetxt(os.path.join(get_dir(opt.outputdir), opt.xp, 'pred_' + opt.datas_order[i] +  '.txt'), d_pred, delimiter=',')
