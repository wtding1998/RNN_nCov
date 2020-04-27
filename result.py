import os
import torch
import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
from rnn_model import *
from utils import normalize, DotDict, Logger, rmse, rmse_tensor, boolean_string, get_dir, get_time, next_dir, get_model, model_dir
from stnn import SaptioTemporalNN


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

    def model_name(self):
        folder_name = os.path.basename(os.path.normpath(self.path))
        return folder_name.split('_')[1]

    def dataset_name(self):
        folder_name = os.path.basename(os.path.normpath(self.path))
        return folder_name.split('_')[0]

    def get_relations(self):
        dataset = self.dataset_name()
        if dataset == "heat":
            data_path = os.path.abspath(os.path.join(self.path, "..", "..", "heat_STNN", "data"))
            relations = np.genfromtxt(os.path.join(data_path, "heat_relations.csv"), delimiter=',', encoding='utf-8-sig')
        else:
            data_path = os.path.abspath(os.path.join(self.path, "..", "..", "disease_STNN", "data"))
            relations = np.genfromtxt(os.path.join(data_path, dataset+"_relations.csv"), delimiter=',', encoding='utf-8-sig')
        relations = torch.tensor(relations).float()
        relations = normalize(relations).unsqueeze(1)
        return relations
        

    def logs(self):
        return get_logs(os.path.join(self.path, self.exp_name))

    def train_loss(self):
        return self.logs()['train_loss.epoch']

    def model(self):
        if self.model_name() == 'LSTM':
            model = LSTMNet(self.config['nx'], self.config['nhid'], self.config['nlayers'], self.config['nx'], self.config['seq_length'])
        if self.model_name() == 'GRU':
            model = GRUNet(self.config['nx'], self.config['nhid'], self.config['nlayers'], self.config['nx'], self.config['seq_length'])
        else:
            model = SaptioTemporalNN(self.get_relations(), self.config['nx'], self.config['checkpoint_interval'], 1, self.config['nz'], self.config['mode'], self.config['nhid'],
                             self.config['nlayers'], self.config['dropout_f'], self.config['dropout_d'],
                             self.config['activation'], self.config['periode'])
        model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt')))
        return model
    
    def pred(self, test_input=None):
        '''
        return pred with (nt, nx)
        if pred_reduce == True, return (nt)
        '''
        pa = os.path.join(self.path, self.exp_name)
        files = os.listdir(pa)
        # ! consider 3-dims output
        pred = []
        for file_name in files:
            if '.txt' in file_name:
                new_pred = np.genfromtxt(os.path.join(self.path, self.exp_name, file_name), delimiter=',')
                if len(new_pred.shape) == 1:
                    new_pred = new_pred[...,np.newaxis]
                pred.append(new_pred)
        pred = np.stack(pred, axis=2)  # (nt_pred, nx, nd)
        if np.isnan(pred).any():
            pred = []
            for file_name in files:
                if '.txt' in file_name:
                    new_pred = np.genfromtxt(os.path.join(self.path, self.exp_name, file_name), delimiter=' ')
                    if len(new_pred.shape) == 1:
                        new_pred = new_pred[...,np.newaxis]
                    pred.append(new_pred)
            pred = np.stack(pred, axis=2)  # (nt_pred, nx, nd)
        return pred

          
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

    def get_df(self, col=['train_loss', 'test_loss', 'true_loss', 'nhid', 'nlayers'], required_list = 'all', mean=False, min=False):
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
        df = pandas.DataFrame(df.values.T, index=df.columns, columns=df.index)[col]
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

        df['used_model'] = df.index
        for i in range(len(df.index)):
            exp_name = df.iloc[i, -1]
            used_model = exp_name.split('-')[0]
            df.iloc[i, -1] = used_model
        return df

    def min_idx(self, col=['test_loss', 'train_loss', 'nhid', 'nlayers'], required_list = 'all'):
        df = self.get_df(col=col, required_list=required_list)
        print("the df is :")
        print(df)
        return df.idxmin()['test_loss']

def plot_pred(pred, data, start_time=0, title='Pred', dim=0):
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

def get_pred(exp_dir, folder):
    pred_dir = {}
    for model_name, exp_name in exp_dir.items():
        pred_dir[model_name] = Exp(exp_name, folder).pred()
    return pred_dir

def plot_pred_by_dir(pred_dir, data, start_time=0, title='Pred', dim=0):
    '''
    pred : {'model_name': (nt_pred, nx, nd)}
    data : (nt, nx, nd)
    '''
    nt_pred = pred_dir[list(pred_dir.keys())[0]].shape[0]
    data_sum = data[:,:, dim].sum(1)
    plotted_dir = {}
    for model_name, pred in pred_dir.items():
        pred_sum = pred[:, :, dim].sum(1) # (nt_pred)
        pred_plotted = np.concatenate([data_sum[-nt_pred - start_time: - nt_pred], pred_sum])
        plotted_dir[model_name] = pred_plotted
    
    data_plotted = data_sum[-nt_pred - start_time:]
    x_axis = np.arange(nt_pred + start_time)
    # plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    plt.grid()
    for model_name, pred_plotted in plotted_dir.items():
        plt.plot(x_axis, pred_plotted, label=model_name, marker='*', linestyle='--')
    plt.plot(x_axis, data_plotted, label='ground_truth', marker='o')
    plt.axvline(x=start_time,ls="--")
    plt.legend()
    plt.title(title)
    return plotted_dir
