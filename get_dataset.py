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

def get_stnn_data(data_dir, disease_name, nt_train, k=1, start_time=0, rescaled_method='d', normalize_method='all', normalize='variance', validation_length=1, relations_names='all', time_datas='all'):
    # get dataset
    opt = DotDict()
    data, opt.datas_order = get_time_data(data_dir, disease_name, start_time, time_datas=time_datas)
    opt.nt, opt.nx, opt.nd = data.size()
    opt.normalize = normalize
    opt.rescaled = rescaled_method
    opt.periode = opt.nt
    relations, opt.relations_order = get_relations(data_dir, disease_name, k, normalize_method=normalize_method, relations_names=relations_names)
    train_data = data[:nt_train]
    # print(train_data.shape)
    # # new_data = data.detach()
    # if rescaled_method == 'd':
    #     opt.mean = []
    #     opt.max = []
    #     opt.min = []
    #     for i in range(opt.nd):
    #         processed_data = new_data[:,:, i]
    #         processed_mean = processed_data.mean().item()
    #         processed_max = processed_data.max().item()
    #         processed_min = processed_data.min().item()
    #         opt.mean.append(processed_mean)
    #         opt.max.append(processed_max)
    #         opt.min.append(processed_min)
    #         new_data[:, :, i] = (processed_data - processed_mean) / (processed_max - processed_min)
    # elif rescaled_method == 'x':
    #     opt.mean = []
    #     opt.max = []
    #     opt.min = []
    #     for i in range(opt.nx):
    #         processed_data = new_data[:, i, :]
    #         processed_mean = processed_data.mean().item()
    #         processed_max = processed_data.max().item()
    #         processed_min = processed_data.min().item()
    #         opt.mean.append(processed_mean)
    #         opt.max.append(processed_max)
    #         opt.min.append(processed_min)
    #         new_data[:, i, :] = (processed_data - processed_mean) / (processed_max - processed_min)
    opt.mean = train_data.mean().item()
    if normalize == 'max_min':
        opt.min = train_data.min().item()
        opt.max = train_data.max().item()
        data = (data - opt.mean) / (opt.max-opt.min)
    elif normalize == 'variance':
        opt.std = torch.std(train_data).item()
        data = (data - opt.mean) / opt.std
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
    train_input = [] # (batch, squence_length, opt.nx*opt.nd)
    train_output = [] # (batch, opt.nx*opt.nd)
    for i in range(nt_train - seq_len):
        new_input = []
        train_input.append(train_data[i:i+seq_len][np.newaxis, ...])
        train_output.append(train_data[i+seq_len][np.newaxis, ...])
    train_input = np.concatenate(train_input, axis=0)
    train_output = np.concatenate(train_output, axis=0)
    test_data = data[nt_train:]
    test_input = data[nt_train - seq_len:nt_train]
    return opt, (train_input, train_output, test_input, test_data)

if __name__ == "__main__":
    # print(get_time_data('data', 'ncov', 0).size())
    print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)[1][0])
    print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)[1][1])
    print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)[1][2])
    print(get_keras_dataset('data', 'jar_increase', 7, 2, start_time=3)[1][3])
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
