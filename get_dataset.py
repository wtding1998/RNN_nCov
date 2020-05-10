import os

import numpy as np
import torch

from utils import DotDict, normalize, normalize_all_row


def get_time_data(data_dir, disease_name, start_time=0, time_datas='all', use_torch=True):
    # data_dir = 'data', disease_name = 'ncov' 
    # return (nt, nx, nd) time series data
    time_data_dir = os.path.join(data_dir, disease_name, 'time_data')
    if not isinstance(time_datas, list):
        time_datas = os.listdir(time_data_dir)
    else:
        time_datas = [data_name+'.csv' for data_name in time_datas]
    data = []
    for time_data in time_datas:
        data_path = os.path.join(time_data_dir, time_data)
        new_data = np.genfromtxt(data_path, encoding='utf-8', delimiter=',')
        if len(new_data.shape) == 1:
            new_data = new_data[..., np.newaxis]
        data.append(new_data)
    data = np.stack(data, axis=2)[start_time:]
    if use_torch:
        return torch.tensor(data).float(), [data_name.replace('.csv', '') for data_name in time_datas]
    else:
        return data.astype(np.float64), [data_name.replace('.csv', '') for data_name in time_datas]

def get_multi_relations(data_dir, disease_name, k, start_time=0):
    '''
    (nx, nrelations, nx, nt)
    '''
    relations_dir = os.path.join(data_dir, disease_name, 'relations')
    relations_names = os.listdir(relations_dir)
    relations = []
    for relations_name in relations_names:
        relations_path = os.path.join(relations_dir, relations_name)
        relations_time = os.listdir(relations_path)
        relation_kind = []
        for t in relations_time:
            relation_path = os.path.join(relations_path, t)
            relation = torch.tensor(np.genfromtxt(relation_path, encoding="utf-8-sig", delimiter=","))
            relation = normalize(relation).unsqueeze(1)
            new_rels = [relation]
            for n in range(k - 1):
                new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relation.size(1))], 1))
            relation = torch.cat(new_rels, 1)
            relation_kind.append(relation)
        relation_kind = torch.stack(relation_kind, dim=3)
        relations.append(relation_kind)
    relations = torch.cat(relations, dim=1)
    return relations.float()[:, :, :, start_time:]

def get_relations(data_dir, disease_name, k, normalize_method='all', relations_names='all'):
    '''
    (nx, nrelations, nx)
    '''
    relations_dir = os.path.join(data_dir, disease_name, 'overall_relations')
    if not isinstance(relations_names, list):
        relations_names = os.listdir(relations_dir)
    else:
        relations_names = [relations_name+'.csv' for relations_name in relations_names]
    relations = []
    for relation_name in relations_names:
        relation_path = os.path.join(relations_dir, relation_name)
        relation = torch.tensor(np.genfromtxt(relation_path, encoding="utf-8-sig", delimiter=","))
        if normalize_method == 'row':
            relation = normalize(relation).unsqueeze(1)
        else:
            relation = normalize_all_row(relation).unsqueeze(1)
        new_rels = [relation]
        for n in range(k - 1):
            new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relation.size(1))], 1))
        relation = torch.cat(new_rels, 1)
        relations.append(relation)
    relations = torch.cat(relations, dim=1)
    return relations.float(), [name.replace('.csv', '') for name in relations_names]

def get_rnn_dataset(data_dir, disease, nt_train, seq_len, start_time=0, normalize='variance'):
    # get dataset
    data = get_time_data(data_dir, disease, start_time)  #(nt, nx, nd)
    # get option
    opt = DotDict()
    opt.nt, opt.nx, opt.nd = data.size()
    opt.normalize = normalize
    opt.mean = data.mean().item()
    if normalize == 'max_min':
        opt.min = data.min().item()
        opt.max = data.max().item()
        data = (data - opt.mean) / (opt.max-opt.min)
    elif normalize == 'variance':
        opt.std = torch.std(data).item()
        data = (data - opt.mean) / opt.std
    # split train / test
    train_input = []
    train_output = []
    for i in range(nt_train - seq_len):
        new_input = []
        for j in range(seq_len):
            new_input.append(data[i+j])
        train_input.append(torch.stack(new_input, dim=0))
        train_output.append(data[i+seq_len])
    train_input = torch.stack(train_input, dim=0)
    train_output = torch.stack(train_output, dim=0)
    test_input = []
    for i in range(seq_len):
        test_input.append(data[nt_train-seq_len+i])
    test_data = data[nt_train:]
    test_input = torch.stack(test_input, dim=0)
    return opt, (train_input, train_output), (test_input, test_data) 

def get_multi_stnn_data(data_dir, disease_name, nt_train, k=1, start_time=0):
    # get dataset
    data = get_time_data(data_dir, disease_name, start_time)
    opt = DotDict()
    opt.nt, opt.nx, opt.nd = data.size()
    opt.periode = opt.nt
    relations = get_multi_relations(data_dir, disease_name, k)
    # ! have to set nt_train = opt.nt - 1
    nt_train = opt.nt - 1
    # make k hop
    # split train / test
    train_data = data[:nt_train]
    test_data = data[nt_train:]
    return opt, (train_data, test_data), relations

def get_stnn_data(data_dir, disease_name, nt_train, k=1, start_time=0, data_normalize='d', relation_normalize='all', normalize='variance', validation_length=1, relations_names='all', time_datas='all'):
    # get dataset
    opt = DotDict()
    data, opt.datas_order = get_time_data(data_dir, disease_name, start_time, time_datas=time_datas)
    opt.nt, opt.nx, opt.nd = data.size()
    opt.normalize = normalize
    opt.data_normalize = data_normalize
    opt.periode = opt.nt
    relations, opt.relations_order = get_relations(data_dir, disease_name, k, normalize_method=relation_normalize, relations_names=relations_names)
    train_data = data[:nt_train]
    if normalize == 'max_min' and data_normalize != 'x':
        opt.mean = train_data.mean().item()
        opt.min = train_data.min().item()
        opt.max = train_data.max().item()
        data = (data - opt.mean) / (opt.max-opt.min)
    elif normalize == 'variance' and data_normalize != 'x':
        opt.mean = train_data.mean().item()
        opt.std = torch.std(train_data).item()
        data = (data - opt.mean) / opt.std
    elif normalize == 'variance' and data_normalize == 'x':
        opt.std = []
        opt.mean = []
        for i in range(opt.nx):
            std = torch.std(train_data[:, i,:]).item()
            if std < 1e-3:
                std = 1
            mean = train_data[:, i,:].mean().item()
            opt.std.append(std)
            opt.mean.append(mean)
            data[:, i,:] = (data[:, i,:] - mean) / std
    elif normalize == 'max_min' and data_normalize == 'x':
        opt.min = []
        opt.max = []
        opt.mean = []
        for i in range(opt.nx):
            mean_value = train_data[:, i,:].mean().item()
            min_value = train_data[:, i,:].min().item()
            max_value = train_data[:, i,:].max().item()
            if (max_value - min_value) < 1e-3:
                max_value = min_value + 1
            opt.min.append(min_value)
            opt.max.append(max_value)
            opt.mean.append(mean_value)
            data[:, i,:] = (data[:, i,:] - mean_value) / (max_value - min_value)

    opt.validation_length = validation_length
    test_data = data[nt_train:]
    train_data = data[:nt_train]
    validation_data = test_data[:opt.validation_length]
    return opt, (train_data, test_data, validation_data), relations

def get_keras_dataset(data_dir, disease_name, nt_train, seq_len, start_time=0, normalize='variance', time_datas=['confirmed'], reduce=True):
    # get dataset
        # data_dir = 'data', disease_name = 'ncov_confirmed' 
    # return (nt, nx, nd) time series data
    opt = DotDict()
    data, opt.datas_order = get_time_data(data_dir, disease_name, start_time=start_time, time_datas=time_datas, use_torch=False)
    opt.nt, opt.nx, opt.nd = data.shape
    opt.reduce = reduce
    if reduce:
        opt.nx = 1
        data = data.sum(1)[:, np.newaxis,:]
    opt.normalize = normalize
    train_data = data[:nt_train]
    opt.mean = np.mean(train_data)
    if normalize == 'max_min':
        opt.min = np.min(train_data)
        opt.max = np.max(train_data)
        data = (data - opt.mean) / (opt.max-opt.min)
    elif normalize == 'variance':
        opt.std = np.std(train_data) * np.sqrt(train_data.size) / np.sqrt(train_data.size-1)
        data = (data - opt.mean) / opt.std
    # split train / test
    data = np.reshape(data, (opt.nt, opt.nx*opt.nd))
    train_data = data[:nt_train]
    data_input = [] # (batch, squence_length, opt.nx*opt.nd)
    data_output = [] # (batch, opt.nx*opt.nd)
    for i in range(opt.nt - seq_len):
        new_input = []
        data_input.append(data[i:i+seq_len][np.newaxis, ...])
        data_output.append(data[i+seq_len][np.newaxis, ...])
    data_input = np.concatenate(data_input, axis=0)
    data_output = np.concatenate(data_output, axis=0)
    train_input = data_input[:nt_train - seq_len]
    train_output = data_output[:nt_train - seq_len]
    val_input = data_input[nt_train - seq_len:]
    val_output = data_output[nt_train - seq_len:]
    test_data = data[nt_train:]
    test_input = data[nt_train - seq_len:nt_train]
    return opt, (train_input, train_output), (val_input, val_output), (test_input, test_data)

def get_true(data, opt):
    true_data = torch.zeros_like(data)
    if opt.normalize == 'max_min' and opt.data_normalize != 'x':
        true_data = data * (opt.max-opt.min) + opt.mean
    elif opt.normalize == 'variance' and opt.data_normalize != 'x':
        true_data = data * opt.std + opt.mean
    elif opt.normalize == 'variance' and opt.data_normalize == 'x':
        for i in range(opt.nx):
            true_data[:, i,:] = data[:, i,:] * opt.std[i] + opt.mean[i]

    elif opt.normalize == 'max_min' and opt.data_normalize == 'x':
        for i in range(opt.nx):
            true_data[:, i,:] = data[:, i,:] * (opt.max[i] - opt.min[i]) + opt.mean[i]
    
    return true_data

if __name__ == "__main__":
    # print(get_time_data('data', 'ncov', 0).size())
    # print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)[1][0])
    # print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)
    # print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)
    # print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)
    # result
    # torch.Size([7, 3, 34, 3])
    # torch.Size([7, 34, 3])
    # torch.Size([3, 34, 3])
    # torch.Size([2, 34, 3])

    # original result
    # torch.Size([97, 3, 29])
    # torch.Size([97, 29])
    # torch.Size([3, 29])
    # torch.Size([56, 29, 1])
    # --- test stnn data ---
    opt, (train_data, test_data, validation_data), relations = get_stnn_data('data', 'mar', 8, data_normalize='x')
    true_train = get_true(train_data, opt)
    test_true = get_true(test_data, opt)
    data = torch.cat([true_train, test_true], dim=0)
    train, _ = get_time_data('data', 'mar')
    print(torch.norm(data - train))
    