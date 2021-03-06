import os
import shutil
import random
import json
import datetime
from collections import defaultdict

import torch
import numpy as np

def rmse(x_pred, x_target, reduce=True):
    if reduce:
        return x_pred.sub(x_target).pow(2).sum(-1).sqrt().mean().item()
    mse = x_pred.sub(x_target).pow(2).sum(2).sqrt().mean(1).squeeze()
    if len(mse.size()) == 0:
        mse =  mse.unsqueeze(0)
    return mse

def rmse_sum_confirmed(x_pred, x_target):
    mse = torch.norm(x_pred.sum(1) - x_target.sum(1)).item()
    return mse

def rmse_np(x_pred, x_target, dim=2):
    '''
    (nt, nx)
    '''
    x_diff = np.power(x_pred - x_target, 2)
    mse = np.mean(np.sqrt(np.sum(x_diff, axis=dim-1)))
    return mse.astype(np.float64)

def rmse_np_like_torch(x_pred, x_target):
    '''
    (nt, nx)
    '''
    x_diff = np.abs(x_pred - x_target)
    mse = np.mean(x_diff)
    return mse.astype(np.float64)

def rmse_matrix(x_pred, x_target):
    '''
    (nt, nx)
    '''
    x_diff = np.sqrt(np.power(x_pred - x_target, 2))
    mse = np.mean(x_diff)
    return mse.astype(np.float64)

def rmse_tensor(x_pred, x_target):
    return x_pred.sub(x_target).pow(2).sum(-1).sqrt().mean()

def copy_nonzero_weights(input):
    """
    Copy the non-zeros weights from a tensor
    Param:
    input: a tensor
    Return a tensor of (#non_zero) 
    """
    non_zero_mask = input.ceil().clamp(0, 1).byte()
    return torch.masked_select(input, non_zero_mask)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def normalize(mx):
    """Row-normalize matrix"""
    rowsum = mx.sum(1)
    r_inv = 1 / rowsum
    r_inv[r_inv == float('Inf')] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv.matmul(mx)
    return mx

def normalize_all_row(mx):
    """2-normalize matrix"""
    nm = torch.norm(mx)
    return mx/nm


def identity(input):
    return input


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Logger(object):
    def __init__(self, log_dir, name, chkpt_interval):
        super(Logger, self).__init__()
        os.makedirs(os.path.join(log_dir, name))
        self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.model_path = os.path.join(log_dir, name, 'model.pt')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        # if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
        #     self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.model_path)      

def get_dir(outputdir):
    return os.path.abspath(os.path.join(os.getcwd(), "..", "output", outputdir))

def get_time():
    return datetime.datetime.now().strftime('%m-%d-%H-%M-%S') + '_' + str(random.random())[3:7]

def time_dir():
    di = {}
    time_list = datetime.datetime.now().strftime('%m-%d-%H-%M-%S').split('-')
    di['hour'] = time_list[0]
    di['minute'] = time_list[1]
    di['day'] = time_list[2]
    di['month'] = time_list[3]
    di['second'] = time_list[4]
    return di
    
def next_dir(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                list.append(h[1])
    return list

def model_dir(outputdir):
    mode_dir = next_dir(outputdir)
    model_dir = {}
    path_dir = {}
    for mode in mode_dir:
        di = os.path.join(outputdir, mode)
        model_list = next_dir(di)  
        model_dir[mode] = model_list
    return model_dir


# get the dataset and model for given folder "dataset_model"
def get_model(folder_name):
    li = folder_name.split('_')
    return li[0], li[1]

def shuffle_list(n, batch_size):
    shuffled_list = list(range(n))
    random.shuffle(shuffled_list)
    return [shuffled_list[i:i + batch_size] for i in range(0, len(shuffled_list), batch_size)]
    
class Logger_keras(object):
    def __init__(self, log_dir, name, chkpt_interval):
        super(Logger_keras, self).__init__()
        os.makedirs(os.path.join(log_dir, name))
        self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.model_path = os.path.join(log_dir, name, 'keras_model.h5')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        # if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
        #     self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        model.save(self.model_path)
        
def get_new_add(dataset):
    '''
    Copy a dataset to dataset_add with time_data changeed into daily increase
    '''
    # --- copy dir ---
    data_path = os.path.join('data', dataset)
    added_data_path = data_path + '_increase'
    if not os.path.exists(added_data_path):
        shutil.copytree(data_path, added_data_path)

    # --- replace timedata with daily increase ---
    time_data_dir = os.path.join(added_data_path, 'time_data')
    data_files = os.listdir(time_data_dir)
    for data_file in data_files:
        data_path = os.path.join(time_data_dir, data_file)
        data = np.genfromtxt(data_path, delimiter=',') # (nt, nx)
        if len(data.shape) == 1:
            data = data[..., np.newaxis]
        daily_add_data = []
        for i in range(data.shape[0]-1):
            daily_add = data[i+1] - data[i]
            daily_add_data.append(daily_add)
        daily_add_data = np.stack(daily_add_data, axis=0)
        print(data_file, 'converted')
        np.savetxt(data_path, daily_add_data, delimiter=',')
    return daily_add_data



if __name__ == "__main__":
    # a = torch.ones(2, 3, 3).float()
    # b = torch.zeros(2, 3, 3).float()
    # print(copy_nonzero_weights(a))
    print(get_new_add('test_rnn'))