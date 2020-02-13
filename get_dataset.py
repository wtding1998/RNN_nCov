import os

import numpy as np
import torch

from utils import DotDict, normalize


def get_time_data(data_dir, disease_name, start_time=0):
    # data_dir = 'data', disease_name = 'ncov' 
    # return (nt, nx, nd) time series data
    time_data_dir = os.path.join(data_dir, disease_name, 'time_data')
    time_datas = os.listdir(time_data_dir)
    data = []
    for time_data in time_datas:
        data_path = os.path.join(time_data_dir, time_data)
        new_data = np.genfromtxt(data_path, encoding='utf-8', delimiter=',')
        new_data = torch.tensor(new_data)
        if len(new_data.size()) == 1:
            new_data = new_data.unsqueeze(1)
        data.append(new_data)

    return torch.stack(data, dim=2).float()[start_time:]

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

def get_relations(data_dir, disease_name, k):
    '''
    (nx, nrelations, nx)
    '''
    relations_dir = os.path.join(data_dir, disease_name, 'overall_relations')
    relations_names = os.listdir(relations_dir)
    relations = []
    for relation_name in relations_names:
        relation_path = os.path.join(relations_dir, relation_name)
        relation = torch.tensor(np.genfromtxt(relation_path, encoding="utf-8-sig", delimiter=","))
        relation = normalize(relation).unsqueeze(1)
        new_rels = [relation]
        for n in range(k - 1):
            new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relation.size(1))], 1))
        relation = torch.cat(new_rels, 1)
        relations.append(relation)
    relations = torch.cat(relations, dim=1)
    return relations.float()

def get_rnn_dataset(data_dir, disease, nt_train, seq_len, start_time=0):
    # get dataset
    data = get_time_data(data_dir, disease, start_time)  #(nt, nx, nd)
    # get option
    opt = DotDict()
    opt.nt, opt.nx, opt.nd = data.size()
    opt.mean = data.mean().item()
    opt.min = data.min().item()
    opt.max = data.max().item()
    data = data - opt.mean
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

def get_stnn_data(data_dir, disease_name, nt_train, k=1, start_time=0):
    # get dataset
    data = get_time_data(data_dir, disease_name, start_time)
    opt = DotDict()
    opt.nt, opt.nx, opt.nd = data.size()
    opt.periode = opt.nt
    relations = get_relations(data_dir, disease_name, k)
    train_data = data[:nt_train]
    opt.mean = train_data.mean().item()
    opt.max = train_data.max().item()
    opt.min = train_data.min().item()
    train_data = (train_data - opt.mean) / (opt.max - opt.min)
    test_data = data[nt_train:]
    test_data = (test_data - opt.mean) / (opt.max - opt.min)

    return opt, (train_data, test_data), relations


if __name__ == "__main__":
    print(get_time_data('data', 'ncov', 0).size())
    # print(get_relations('data', 'ncov', 1))
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
