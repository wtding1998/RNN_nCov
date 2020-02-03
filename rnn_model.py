import numpy as np
import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, seq_length):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(input_size, hid_size, hid_layers, batch_first = True)
        self.linear = nn.Linear(hid_size, output_size, bias=False)
        self.out = nn.Linear(seq_length, 1, bias=False)
        self.input_size = input_size
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.seq_length = seq_length

    def forward(self, input):
        nd = input.size(-1)
        nx = input.size(-2)
        input = input.contiguous()
        input = input.view(-1, self.seq_length, nd*nx)
        out, _ = self.rnn(input)
        outs = []
        for timestep in range(input.size(1)):
            outs.append(torch.relu(self.linear(out[:,timestep,:])))
        new_out = torch.stack(outs, dim=2)
        outs = []
        for batch in range(input.size(0)):
            outs.append(torch.relu(self.out(new_out[batch])).view(-1,1))
        output = torch.stack(outs, dim=0).contiguous()
        output = output.view(-1, nx, nd)
        return output
        
    def update(self, init, pred):
        # init : seq_len , nx, nd
        nd = init.size(-1)
        nx = init.size(-2)
        new_seq = []
        for i in range(self.seq_length-1):
            new_seq.append(init[i+1])
        new_seq.append(pred.squeeze(0))
        # print('pred:', pred.size())
        # print('init:', init[0].size())
        return torch.stack(new_seq, dim=0) # seq_len, nx, nd
    
    def generate(self, init, length):
        nd = init.size(-1)
        nx = init.size(-2)
        pred_list = []
        for i in range(length):
            new_pred = self.forward(init.unsqueeze(0)) # 1, nx, nd
            pred_list.append(new_pred)
            init = self.update(init, new_pred)
        return torch.cat(pred_list, dim=0)

class GRUNet(nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, seq_length):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(input_size, hid_size, hid_layers, batch_first = True)
        self.linear = nn.Linear(hid_size, output_size, bias=False)
        self.out = nn.Linear(seq_length, 1, bias=False)
        self.input_size = input_size
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.seq_length = seq_length

    def forward(self, input):
        out, _ = self.rnn(input)
        outs = []
        for timestep in range(input.size(1)):
            outs.append(torch.relu(self.linear(out[:,timestep,:])))
        new_out = torch.stack(outs, dim=2)
        outs = []
        for batch in range(input.size(0)):
            outs.append(torch.relu(self.out(new_out[batch])).view(-1,1))
        return torch.stack(outs, dim=0).view(-1, self.input_size)
    
    def update(self, init):
        pred = self.forward(init).view(-1)
        new_seq = []
        for i in range(self.seq_length-1):
            new_seq.append(init[0, i+1])
        new_seq.append(pred)
        return torch.stack(new_seq, dim=0).unsqueeze(0)
    
    def generate(self, init, length):
        init = init.unsqueeze(0)
        pred_list = []
        for i in range(length):
            pred_list.append(self.forward(init))
            init = self.update(init)
        return torch.stack(pred_list, dim=0).view(length, self.input_size, 1)
