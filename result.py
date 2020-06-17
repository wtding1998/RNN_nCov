import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

from get_dataset import get_relations, get_time_data, get_true
from rnn_model import *
from stnn import (SaptioTemporalNN_concat, SaptioTemporalNN_input,
                  SaptioTemporalNN_input_simple, SaptioTemporalNN_v0)
from utils import (DotDict, Logger, boolean_string, get_dir, get_model,
                   get_time, model_dir, next_dir, normalize, rmse, rmse_np,
                   rmse_np_like_torch, rmse_tensor)
from generate_scr import output_one

# from keras.models import load_model


def get_config(model_dir):
    # get config
    with open(os.path.join(model_dir, 'config.json')) as f:
        config_logs = json.load(f)
    # for opt in print_list:
    #     print(config_logs[opt])
    # print("the test loss for %s is : %f" %(model_dir, config_logs['test_loss']))
    return config_logs

def get_logs(model_dir):
    # get logs
    with open(os.path.join(model_dir, 'logs.json')) as f:
        logs = json.load(f)
    return logs

# print the information for all model of the given folder | aids_LSTM

def get_list(string, folder):
    model_list = next_dir(folder)
    li = []
    for i in model_list:
        if string in i:
            li.append(i)
    return li


def get_df(folder, col=['test_loss', 'train_loss', 'true_loss', 'nhid', 'nlayers'], required_list = 'all'):
    if isinstance(required_list, str):
        required_list = os.listdir(folder)
    # df_list = []
    # for model_name in required_list: 
    #     config = get_config(os.path.join(folder, model_name))
    #     new_df = pandas.DataFrame([config])[col]
    #     new_df.index = [model_name]
    #     df_list.append(new_df)
    # df =  pandas.concat(df_list, join='outer')
    # df.name = folder.split('/')[-1]
    df_dir = {}
    for exp_name in required_list:
        try:
            config = get_config(os.path.join(folder, exp_name))
            df_dir[exp_name] = config
        except:
            continue
    df = pandas.DataFrame(df_dir)
    return df


class Exp():
    def __init__(self, exp_name, path):
        self.path = path
        self.exp_name = exp_name
        self.config = get_config(os.path.join(self.path, self.exp_name))
        self.model_name = exp_name.split('-')[0]
        if self.model_name == 'keras':
            self.model_name == self.config['rnn_model']
        self.nt = self.config['nt']
        self.nx = self.config['nx']
        self.nd = self.config['nd']
        self.nz = self.config.get('nz')
        self.nt_train = self.config['nt_train']
        if 'increase' in self.config and self.config['increase']:
            self.increase = True
            self.get_pred_by_increase()
        else:
            self.increase = False
        # self.calculate_rmse_loss()
        self.config['sum_loss'] = self.pred_loss()
        # replace name
        if 'final_rmse_score' not in self.config.keys():
            self.config['final_rmse_score'] = self.config.get('final_rmse_loss', None)

        if 'final_sum_score' not in self.config.keys():
            self.config['final_sum_score'] = self.config.get('final_sum_loss', None)

        if ('normalize_method' not in self.config.keys()) and ('relation_normalize' in self.config.keys()):
            self.config['normalize_method'] = self.config['relation_normalize']
        with open(os.path.join(self.path, self.exp_name, 'config.json'), 'w') as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

    def dataset_name(self):
        folder_name = os.path.basename(os.path.normpath(self.path))
        return folder_name.split('_')[0]

    def relations(self):
        relations, _ = get_relations(self.config['datadir'], self.config['dataset'], self.config['khop'], self.config['normalize_method'], self.config['relations_order'])
        return relations
        
    def logs(self):
        return get_logs(os.path.join(self.path, self.exp_name))

    def train_loss(self):
        return self.logs()['train_loss.epoch']

    def model(self):
        # if (self.model_name == 'LSTM') or (self.model_name == 'GRU'):
        #     model = load_model(os.path.join(self.path, self.exp_name, 'model.h5'))

        if self.model_name == 'ori':
            model = SaptioTemporalNN_v0(self.relations(), self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                        self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt')))

        if self.model_name == 'v1':
            model = SaptioTemporalNN_concat(self.relations(), self.data_torch()[:self.config['nt_train']], self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                        self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt')))
            
        if self.model_name == 'v2':
            model = SaptioTemporalNN_input(self.relations(), self.data_torch()[:self.config['nt_train']], self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                        self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'], self.config['simple_dec'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt')))

        if self.model_name == 'v3':
            model = SaptioTemporalNN_input_simple(self.relations(), self.data_torch()[:self.config['nt_train']], self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                        self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt')))

        return model

    def data_torch(self, increase=False):
        if self.increase:
            dataset = self.config['dataset'].replace('_increase', '')
        else:
            dataset = self.config['dataset']

        if increase:
            if 'time_datas' in self.config.keys():
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset+'_increase', start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas=self.config['time_datas'], use_torch=True)
            else:
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset+'_increase', start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas='all', use_torch=True)
        else:
            if 'time_datas' in self.config.keys():
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset, start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas=self.config['time_datas'], use_torch=True)
            else:
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset, start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas='all', use_torch=True)
        return data

    def data_np(self, increase=False):
        if self.increase:
            dataset = self.config['dataset'].replace('_increase', '')
        else:
            dataset = self.config['dataset']

        if increase:
            if 'time_datas' in self.config.keys():
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset + '_increase', start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas=self.config['time_datas'], use_torch=False)
            else:
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset + '_increase', start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas='all', use_torch=False)
        else:
            if 'time_datas' in self.config.keys():
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset, start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas=self.config['time_datas'], use_torch=False)
            else:
                data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset, start_time=self.config['start_time'], delete_time=self.config.get('delete_time', 0), time_datas='all', use_torch=False)
        return data
    
    def pred_loss(self, reduce=True, increase=False):
        data = self.data_np(increase)
        if not increase and hasattr(self, 'nt_pred'):
            nt_train = self.nt - self.nt_pred + 1
        else:
            nt_train = self.nt_train
        test_data = data[nt_train:,:, 0]
        pred = self.pred()[:,:, 0]
        if reduce:
            test_data = test_data.sum(1)
            pred = pred.sum(1)
            return np.linalg.norm(test_data - pred) / (self.nt - nt_train)
        else:
            return rmse_np(test_data, pred)

    def pred(self, increase=False):
        '''
        return pred with (nt, nx)
        if pred_reduce == True, return (nt)
        '''
        pa = os.path.join(self.path, self.exp_name)
        files = os.listdir(pa)
        # ! consider 3-dims output
        pred = []
        if increase:
            prefix = 'increase'
        else:
            prefix = 'pred'
        for file_name in files:
            if prefix in file_name:
                new_pred = np.genfromtxt(os.path.join(self.path, self.exp_name, file_name), delimiter=',')
                if len(new_pred.shape) == 1:
                    new_pred = new_pred[...,np.newaxis]
                pred.append(new_pred)
        pred = np.stack(pred, axis=2)  # (nt_pred, nx, nd)
        if np.isnan(pred).any():
            pred = []
            for file_name in files:
                if prefix in file_name:
                    new_pred = np.genfromtxt(os.path.join(self.path, self.exp_name, file_name), delimiter=' ')
                    if len(new_pred.shape) == 1:
                        new_pred = new_pred[...,np.newaxis]
                    pred.append(new_pred)
            pred = np.stack(pred, axis=2)  # (nt_pred, nx, nd)
        return pred
    
    def calculate_rmse_loss(self, data_kind='confirmed'):
        '''
        Calculate the data_loss of exp
        '''
        nd = self.config['nd']
        nx = self.config['nx']
        # --- get test data ---
        data = self.data_np()
        # --- get prediction ---
        pred_data = self.pred()
        nt_pred = pred_data.shape[0]
        test_data = data[-nt_pred:]
        # if nd = 1, then the pred data is confirmed
        if nd == 1:
            pred_data.shape = (-1, nx)
            test_data.shape = (-1, nx)
        # if nd > 1, then the pred could be pred_001 or pred_confirmed, if datas_order in config
        # elif 'datas_order' in self.config:
        #     data_index = self.config['datas_order'].index(data_kind)
        #     pred_data = pred_data[:,:, data_index]
        #     test_data = test_data[:,:, data_index]
        # or else is in data_kind
        else:
            pred_data = pred_data[:,:, 0]
            test_data = test_data[:,:, 0]
        # print(test_data)
        # print(pred_data)
        rmse_loss = rmse_np(pred_data, test_data)
        self.config['loss_true_' + data_kind] = rmse_loss
        # --- calculate loss before renormalize---
        if self.config['normalize'] == 'variance':
            self.config['loss_' + data_kind] = rmse_loss / self.config['std']
        if self.config['normalize'] == 'min_max':
            self.config['loss_' + data_kind] = rmse_loss / (self.config['max'] - self.config['min'])
        # if self.config['model'] == 'v3':
        #     self.config['model'] = 'v1'
        if self.config['mode'] == None:
            self.config['mode'] = 'default'
        return rmse_loss

    def validation_loss(self, torch_like=False):
        # --- validation data ---
        data = self.data_np()
        validation_length = self.config['validation_length']
        validation_data = data[self.nt_train:self.nt_train + validation_length]
        pred = self.pred()[:validation_length]
        if torch_like:
            loss = rmse_np_like_torch(validation_data, pred) / self.config['std']
        else:
            loss = np.linalg.norm(validation_data.sum(1) - pred.sum(1)) / validation_length
        return loss

    def draw_loss(self, ylim1=0, ylim2=1, logs=['sum']):
        log = self.logs()
        di = {}
        rmse_loss = log['test_epoch.rmse']
        sum_loss = log['test_epoch.sum']
        dyn_loss = log['train_epoch.loss_dyn']
        dec_loss = log['train_epoch.mse_dec']
        x = np.arange(log['epoch'])
        if 'rmse' in logs:
            plt.plot(x, rmse_loss, label='rmse')
        if 'sum' in logs:
            plt.plot(x, sum_loss, label='sum')
        if 'dyn' in log:
            plt.plot(x, dyn_loss, label='dyn')
        if 'dec' in logs:
            plt.plot(x, dec_loss, label='dec')
        plt.legend()
        plt.ylim(ylim1, ylim2)
        
    
    def plot_relations(self):
        relations = self.config['relations_order']
        logs = self.logs()
        nepoch = self.config['nepoch']
        epochs = np.arange(nepoch)
        # relations_result_dir = {}
        for i, relation in enumerate(relations):
            max_list = logs["train_epoch." + relation + "_max"]
            min_list = logs["train_epoch." + relation + "_min"]
            mean_list = logs["train_epoch." + relation + "_mean"]
            plt.title(relation + ' change ')
            plt.plot(epochs, max_list, label=relation + '_max')
            plt.plot(epochs, min_list, label=relation+'_min')
            plt.plot(epochs, mean_list, label=relation + '_mean')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()

            # relations_result_dir[relation] = [max_list, min_list, mean_list]
    def get_pred_by_increase(self):
        '''
        get pred_data by increase_data and data
        '''
        # --- calculate prediction ---
        print('generate prediction for', self.exp_name)
        data, datas_name = get_time_data('data', self.config['dataset'].replace('_increase', ''), start_time=self.config['start_time'], delete_time=self.config.get('end_time', 0), time_datas=self.config.get('time_datas', ['confirmed']), use_torch=False) 
        increase_data = self.pred(increase=True) # (nt_pred, 1, nd)
        nt_pred = increase_data.shape[0]
        # --- add nt_pred in config ---
        self.config['nt_pred'] = nt_pred
        self.nt_pred = nt_pred
        # self.config['nt_train'] = self.config['nt'] - nt_pred
        # self.nt_train = self.config['nt'] - nt_pred

        if self.nx == 1:
            data = np.reshape(data.sum(1), (self.nt + 1, 1, -1)) # (nt, 1, nd)
            increase_data = np.reshape(increase_data, (nt_pred, 1, -1))

        start_data = data[-nt_pred - 1] # (nx, nd)
        pred_data = np.zeros(increase_data.shape)
        pred_data[0] = start_data + increase_data[0]
        for t in range(nt_pred-1):
            pred_data[t + 1] = pred_data[t] + increase_data[t + 1]

        # --- save prediction ---
        for i, data_name in enumerate(datas_name):
            np.savetxt(os.path.join(self.path, self.exp_name, 'pred_'+data_name+'.txt'), pred_data[:,:,i], delimiter=',')

        # --- calculate sum loss for confirmed---
        confirmed_pred = pred_data[:,:, 0].sum(1)
        confirmed_test = data[-nt_pred:,:, 0].sum(1)
        self.config['sum_loss'] = np.linalg.norm(confirmed_pred - confirmed_test) / nt_pred
        return pred_data

    def train_pred(self, pred_test=False):
        '''
        get the pred for the train_time
        '''
        model = self.model()
        factors = model.factors
        train_pred = []
        for i in range(self.nt_train):
            train_pred.append(model.decode_z(factors[i]))
        train_pred = torch.stack(train_pred, dim=0)
        return self.rescaled(train_pred.detach().numpy())
    
    def rescaled(self, data):
        '''
        convert the scaled data into original scale
        '''
        normalize_config = {}
        normalize_config['normalize'] = self.config['normalize']
        normalize_config['nx'] = self.config['nx']
        normalize_config['std'] = self.config.get('std', None)
        normalize_config['mean'] = self.config.get('mean', None)
        normalize_config['min'] = self.config.get('min', None)
        normalize_config['max'] = self.config.get('max', None)
        if 'data_normalize' not in self.config.keys():
            normalize_config['data_normalize'] = 'd'
        else:
            normalize_config['data_normalize'] = self.config['data_normalize']

        if normalize_config['data_normalize'] == 'x':
            data = np.reshape(data, (-1, self.nx, self.nd))

        # if self.config['normalize'] == 'variance' and self.config.get('data_normalize', 'd') == 'd':
        #     true_data = data * self.config['std'] + self.config['mean']
        # elif self.config['normalize'] == 'min_max' and self.config.get('data_normalize', 'd') == 'd':
        #     true_data = data * (self.config['max'] - self.config['min']) + self.config['mean']
        # elif self.config['normalize'] == 'variance' and self.config.get('data_normalize', 'd') == 'x':
        return get_true(data, normalize_config, use_torch=False)

    def generate(self, nsteps, reduce=False, axis=0):
        model = self.model()
        x, z = model.generate(nsteps)
        x = x[:, :, axis].detach().numpy()
        z = z.detach().numpy()
        # x = x.sum(1)
        return self.rescaled(x), z

    def plot_distribution(self, start_time=0):
        # --- get data ---
        data = self.data_np(increase=False)
        pred = self.pred(increase=False)
        data = data[:,:,0]
        pred = pred[:,:,0]
        # --- concat data from start_time ---
        nt_pred = pred.shape[0]
        if start_time >= 0:
            data = data[-nt_pred - start_time:]
            pred = np.concatenate([data[:start_time], pred], axis=0)
        else:
            pred = np.concatenate([data[-nt_pred:], pred], axis=0)
        
        error = np.abs(data - pred)
        # --- show image ---
        plt.subplot(1, 3, 1)
        plt.imshow(data.T)
        plt.title('Data')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(pred.T)
        plt.title('Pred')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(error.T)
        plt.title('Error')
        plt.colorbar()

class Printer():
    def __init__(self, folder):
        self.folder = folder
        self.dataset = self.folder.split('_')[0]
        self.model = self.folder.split('_')[1]

    def exp_models(self):
        return next_dir(self.folder)

    def get_model(self, string):
        model_list = next_dir(self.folder)
        li = []
        for i in model_list:
            if string in i:
                li.append(i)
        return li

    def get_df(self, col=['train_loss', 'test_loss', 'true_loss', 'nhid', 'nlayers'], required_list = 'all', mean=False, min=False, increase=False, nt_train=0):
        if isinstance(required_list, str):
            required_list = next_dir(self.folder)
        df_dir = {}
        for exp_name in required_list:
            try:
                config = get_config(os.path.join(self.folder, exp_name))
                df_dir[exp_name] = config
            except:
                print(exp_name, ' x')

        df = pandas.DataFrame(df_dir)
        df = pandas.DataFrame(df.values.T, index=df.columns, columns=df.index)
        if nt_train > 0:
            df = df.loc[df['nt_train'] == nt_train]
        if 'increase' in df.columns:
            if increase:
                df = df.loc[df['increase'] == True]
            else:
                df = df.loc[df['increase'] == False]

        df['used_model'] = df.index
        for i in range(len(df.index)):
            exp_name = df.iloc[i, -1]
            used_model = exp_name.split('-')[0]
            df.iloc[i, -1] = used_model

        df = df[col]
        # df_list = []
        # for model_name in required_list: 
        #     config = get_config(os.path.join(self.folder, model_name))
        #     new_df = pandas.DataFrame([config])[col]
        #     new_df.index = [model_name]
        #     df_list.append(new_df)
        # df =  pandas.concat(df_list, join='outer')
        if mean:
            df.loc['mean'] = df.apply(lambda x: x.mean())
        if min:
            df.loc['min'] = df.apply(lambda x: x.min())



        return df

    def min_idx(self, col=['test_loss', 'train_loss', 'nhid', 'nlayers'], required_list = 'all'):
        df = self.get_df(col=col, required_list=required_list)
        print("the df is :")
        print(df)
        return df.idxmin()['test_loss']

def plot_pred(pred, data, nt_pred_tim=0, title='Pred', dim=0):
    '''
    pred : (nt_pred, nx)
    data : (nt, nx)
    '''
    pred_sum = pred[:, :, dim].sum(1) # (nt_pred)
    data_sum = data[:, :, dim].sum(1) # (nt)
    nt_pred = pred_sum.shape[0]
    data_plotted = data_sum[-nt_pred - start_time:]
    pred_plotted = np.concatenate([data_sum[-nt_pred - start_time: - nt_pred], pred_sum])
    x_axis = np.arange(nt_pred + start_time)

    plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    plt.grid()
    plt.plot(x_axis, pred_plotted, label='预测值', marker='*', linestyle='--')
    plt.plot(x_axis, data_plotted, label='真实值', marker='o')
    plt.axvline(x=start_time,ls="--")
    plt.legend()
    plt.title(title)

def sorted_by_loss(df, value='test_loss', merges=['used_model']):
    # get the different kind of model
    models = []
    models_loss = {}
    for i in range(len(df.index)):
        exp_data = df.iloc[i]
        merged_data = []
        for kind in merges:
            merged_data.append(exp_data[kind])

        loss = exp_data[value]
        merged_data = [str(a) for a in merged_data]
        merged_data = '_'.join(merged_data) 

        if merged_data not in models:
            models.append(merged_data)
            models_loss[merged_data] = [loss, i]
        elif loss < models_loss[merged_data][0]:
            models_loss[merged_data] = [loss, i]
    indexs = []
    for k, v in models_loss.items():
        indexs.append(v[1])
    return df.iloc[indexs]

def get_exp_name(df, choosen_values=['used_model']):
    index_dir = {}
    for i in range(len(df.index)):
        exp_data = df.iloc[i]
        model_index = []
        for value in choosen_values:
            model_index.append(exp_data[value])
        model_index = [str(value) for value in model_index]
        model_index = '_'.join(model_index)
        index_dir[model_index] = df.index[i]
    return index_dir

def get_pred(exp_dir, folder, train=False):
    pred_dir = {}
    for model_name, exp_name in exp_dir.items():
        exp = Exp(exp_name, folder)
        pred_data = exp.pred()
        if train:
            train_pred = exp.train_pred()
            pred_data = np.concatenate([train_pred, pred_data], axis=0)
        pred_dir[model_name] = pred_data
    data = exp.data_np()
    return pred_dir, data

def plot_pred_by_dir(exp_dir, folder, line_time=0, title='Pred', dim=0, train=False, increase=False):
    '''
    pred : {'model_name': (nt_pred, nx, nd)}
    data : (nt, nx, nd)
    '''
    pred_dir = {}
    loss_dir = {}
    for model_name, exp_name in exp_dir.items():
        exp = Exp(exp_name, folder)
        pred_data = exp.pred(increase)
        nt_pred = pred_data.shape[0]
        if train:
            train_pred = exp.train_pred()
            train_pred = np.reshape(train_pred, (-1, pred_data.shape[1], pred_data.shape[2]))
            pred_data = np.concatenate([train_pred, pred_data], axis=0)
        pred_dir[model_name] = pred_data
    data = exp.data_np(increase)
    print(exp.config['dataset'])
    data_sum = data[:,:, dim].sum(1)
    plotted_dir = {}
    if line_time < 0:
        line_time = data_sum.shape[0] - nt_pred
    for model_name, pred in pred_dir.items():
        pred_sum = pred[:, :, dim].sum(1) # (nt_pred)
        if not train:
            pred_sum = np.concatenate([data_sum[-nt_pred - line_time: - nt_pred], pred_sum])
        plotted_dir[model_name] = pred_sum
    if not train:
        data_sum = data_sum[-nt_pred - line_time:]
        x_axis = np.arange(nt_pred + line_time)
    else:
        nt = data_sum.shape[0]
        x_axis = np.arange(nt)
        line_time = nt - nt_pred -1
    # plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    plt.grid()
    for model_name, pred_sum in plotted_dir.items():
        print(np.linalg.norm(pred_sum - data_sum) / nt_pred)
        plt.plot(x_axis, pred_sum, label=model_name, marker='*', linestyle='--')
    plt.plot(x_axis, data_sum, label='ground_truth', marker='o')
    plt.axvline(x=line_time,ls="--")
    plt.legend()
    plt.title(title)
    plt.show()
    return plotted_dir, data_sum

def output_scr_by_dir(di, dir_path, minepoch='sum', write='w', model='stnn', configs=['test', 'activation', 'batch_size', 'dataset', 'increase', 'lambd', 'lr', 'manualSeed', 'mode', 'nhid', 'nlayers', 'nt_train', 'data_normalize', 'nz', 'sch_bound', 'start_time', 'validation_length', 'test', 'time_datas']):
    if model == 'rnn':
         configs=['rnn_model', 'activation', 'batch_size', 'dataset', 'increase', 'lr', 'manualSeed', 'nhid', 'nlayers', 'nt_train', 'start_time', 'seq_length']
    with open(r'small_dir.txt', write) as f:
        for model_name, exp_name in di.items():
            config_di = {}
            exp_config = get_config(os.path.join(dir_path, exp_name))
            config_di['model'] = model_name
            # --- get useful info ---
            for config in configs:
                config_di[config] = exp_config[config] 
            
            # --- get epoch ---
            config_di['nepoch'] = exp_config['min_' + minepoch + '_epoch'] + 1
            # --- output command ---
            output_one(config_di, f)
            print(file=f)

def process_config(folder):
    exp_names = os.listdir(folder)
    for exp_name in exp_names:
        exp = Exp(exp_name, folder)
        print(exp_name)
    
if __name__ == "__main__":
    # === Test ===


    # --- test calculate_rmse_loss ---
    # path = 'D:/Jupyter_Documents/ML-code/research_code/output/jar0426'
    # exp_name = 'v3-stnn_04-27-08-31-14'
    # exp_name = 'v3-stnn_04-27-09-19-04'
    # exp = Exp(exp_name, path)

    # print(exp.config['score_true_confirmed'])
    # print(exp.calculate_rmse_loss())

    # --- test plot relation change ---
    # path = 'D:/Jupyter_Documents/ML-code/research_code/output/test'
    # exp_name = 'v2-stnn_04-27-11-32-30'
    # exp = Exp(exp_name, path)
    # exp.plot_relations()

    # --- test increase data ---
    path = 'D:/Jupyter_Documents/ML-code/research_code/output/test_fli'
    exp_name = 'ori-stnn_00-06-05-04-56'
    exp_dir = {'test': exp_name}
    # exp_name = 'v2-stnn_05-03-00-05-59_0251'
    # exp = Exp(exp_name, path)
    # print(exp.plot_train_times().shape)
    # pred_dir,data = get_pred(exp_dir, path, train=False)
    plot_pred_by_dir(exp_dir, path, increase=False)
