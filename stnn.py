import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from module import MLP
from utils import identity


class SaptioTemporalNN_multitime(nn.Module):
    def __init__(self,
                 relations,
                 nx,
                 nt_train,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1):
        super(SaptioTemporalNN_multitime, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt_train = nt_train
        self.nx = nx
        self.nz = nz
        self.nd = nd
        self.mode = mode
        # kernel
        self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        eye = torch.stack([
            torch.eye(nx).to(device).unsqueeze(1) for i in range(nt_train + 1)
        ],
                          dim=3)
        if mode is None or mode == 'refine':
            self.relations = torch.cat((eye, relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat((eye, torch.ones(nx, 1, nx).to(device)),
                                       1)
        self.nr = self.relations.size(1)
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.Tensor(nt_train, nx, nz))
        self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
        self.decoder = nn.Linear(nz, nd, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() -
                             self.nx * self.nt_train))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx, nt_train))
        # init
        self._init_weights(periode)

    def _init_weights(self, periode):
        initrange = 0.1
        if periode >= self.nt_train:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(
                    -initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self, t):
        if self.mode is None:
            return self.relations[:, :, :, t]
        else:
            weights = F.hardtanh(self.rel_weights[:, :, :, t], 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights[:, :, :, t].new(
                    self.nx, self.nx).copy_(self.relations[:, 0, :,
                                                           t]).unsqueeze(1)
                inter = self.rel_weights[:, :, :,
                                         t].new_zeros(self.nx, self.nr - 1,
                                                      self.nx)
                inter.masked_scatter_(self.relations[:, 1:, :, t], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0, :, t].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z, t):
        z_context = self.get_relations(t).matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        z_gen = []
        for t in t_idx:
            rels = self.get_relations(t)
            z_input = self.drop(self.factors[t])
            z_context = rels[x_idx].matmul(z_input).view(-1, self.nr * self.nz)
            z_gen.append(self.dynamic(z_context))
        z_gen = torch.stack(z_gen, dim=0)
        return self.activation(z_gen)

    def generate(self, nsteps, start_time):
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z, start_time + t)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights


class SaptioTemporalNN(nn.Module):
    def __init__(self,
                 relations,
                 nx,
                 nt,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1):
        super(SaptioTemporalNN, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'refine':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), torch.ones(
                    nx, 1, nx).to(device)), 1)
        self.nr = self.relations.size(1) # number of relations, here nr = 2
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.randn(nt, nx, nz))
        self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
        self.decoder = nn.Linear(nz, nd, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx))
        # init
        self._init_weights(periode)

    def _init_weights(self, periode): #初始化权重
        initrange = 1.0
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(
                    -initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(
                    self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1,
                                                   self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        z_context = rels[x_idx].matmul(z_input).view(-1,
                                                     self.nr * self.nz) ## ?
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps):
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights


class SaptioTemporalNN_noz(nn.Module):
    def __init__(self,
                 relations,
                 nx,
                 nt,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1):
        super(SaptioTemporalNN_noz, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        nz = nd
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'refine':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), torch.ones(
                    nx, 1, nx).to(device)), 1)
        self.nr = self.relations.size(1) # number of relations, here nr = 2
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = torch.randn(nt, nx, nz)
        self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
        self.decoder = nn.Linear(nz, nd, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx))
        # init
        #  self._init_weights(periode)

    def _init_weights(self, periode): #初始化权重
        initrange = 1.0
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(
                    -initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(
                    self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1,
                                                   self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        z_context = rels[x_idx].matmul(z_input).view(-1,
                                                     self.nr * self.nz) ## ?
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps):
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights

class SaptioTemporalNN_large(nn.Module):
    def __init__(self,
                 relations,
                 nx,
                 nt,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1):
        super(SaptioTemporalNN_large, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.nd = nd
        self.mode = mode
        # kernel
        self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'refine':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), torch.ones(
                    nx, 1, nx).to(device)), 1)
        self.nr = self.relations.size(1) # number of relations, here nr = 2
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.randn(nt, nx, nz))
        self.dynamic = MLP(nz * self.nr * nx, nhid, nz * nx, nlayers, dropout_d)
        self.decoder = nn.Linear(nz * nx, nd * nx, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx))
        # init
        self._init_weights(periode)

    def _init_weights(self, periode): #初始化权重
        initrange = 1.0
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(
                    -initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(
                    self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1,
                                                   self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        '''
        original : z(batch, self.nz)
        new : z(batch, self.nx * self.nz)
        '''
        rels = self.get_relations()
        z_context = []
        for r in range(rels.size(1)):
            z_context.append(rels[:, r].matmul(z))
        z_context = torch.cat(z_context, dim=0).view(-1, self.nr * self.nz * self.nx)
        # z_context = self.get_relations().matmul(z)
        # z_context = z_context.view(-1, self.nr * self.nz * self.nx)
        z_next = self.dynamic(z_context)
        return self.activation(z_next).view(self.nx, self.nz)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx):
        z_inf = self.drop(self.factors[t_idx])
        z_inf = z_inf.contiguous()
        z_inf = z_inf.view(-1, self.nz * self.nx)
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])  # (nt, nx, nz)
        z_context = []
        for t in range(z_input.size(0)):
            z_t = []
            for r in range(rels.size(1)):
                z_t.append(rels[:, r].matmul(z_input[t]))
            z_t = torch.cat(z_t, dim=0)
            z_context.append(z_t)
        z_context = torch.stack(z_context, dim=0).view(-1, self.nr * self.nz * self.nx)
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps):
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen).view(-1, self.nx * self.nz)
        x_gen = self.decode_z(z_gen).view(-1, self.nx, self.nd)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights