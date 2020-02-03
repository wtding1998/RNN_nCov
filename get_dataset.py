import os

import numpy as np
import torch

from utils import DotDict, normalize


def get_time_data(data_dir, disease_name):
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

    return torch.stack(data, dim=2).float()


def get_rnn_dataset(data_dir, disease, nt_train, seq_len):
    # get dataset
    data = get_time_data(data_dir, disease) #(nt, nx, nd)
    # get option
    opt = DotDict()
    opt.nt, opt.nx, opt.nd = data.size()
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

if __name__ == "__main__":
    opt, (train_input, train_output), (test_input, test_output) = get_rnn_dataset('data', 'ncov', 10, 3)
    print(train_input.size())
    print(train_output.size())
    print(test_input.size())
    print(test_output.size())
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