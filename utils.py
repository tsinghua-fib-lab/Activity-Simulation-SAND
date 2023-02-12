import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from bisect import bisect_right
from config import *
import pandas as pd
import sys
import bisect


def distribution_cal(x,cate):
    cal = pd.value_counts(x,normalize=True)
    result = []
    for i in cate:
        result.append(cal[i] if i in cal.keys() else 0)
    return result

def arr_to_distribution(arr, bins_info):
    bins = len(bins_info)+1
    distribution, _ = np.histogram(np.array(arr),bins = bins,range=(0,bins_info[-1]))
    return distribution
    

# SoftPlus activation function add epsilon
class SoftPlus(nn.Module):

    def __init__(self, beta=1.0, threshold=20, epsilon=1.0e-15, dim=None):
        super(SoftPlus, self).__init__()
        self.Softplus = nn.Softplus(beta, threshold)
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, x):
        # apply softplus to first dim dimension
        if self.dim is None:
            result = self.Softplus(x) + self.epsilon
        else:
            result = torch.cat((self.Softplus(x[..., :self.dim])+self.epsilon, x[..., self.dim:]), dim=-1)

        return result


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0.0, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)
        
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)
        return x

class MLP(nn.Module):

    def __init__(self, dim_in,  dim_hidden, dim_out, num_hidden=0, activation=nn.CELU()):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)


def t2grid(time):
    return bisect_right(config.time_grid,time)

# This function need to be stateless
class ODEJumpFunc(nn.Module):

    def __init__(self, config, args, device, activation=nn.CELU()):
        super(ODEJumpFunc, self).__init__()

        self.jump_type='simulate'
        self.config = config
        self.args = args
        self.device = device
        self.evnts = []

        self.F = nn.ModuleList([MLP(self.args.dim_c+self.args.dim_h+self.args.hour_dim+self.args.week_dim, self.args.dim_hidden,self.args.dim_c,self.args.num_hidden, activation) for i in range(3)])

        self.hour_emb = nn.Embedding(
                num_embeddings=self.config.total_hours, embedding_dim=self.args.hour_dim, scale_grad_by_freq=False, sparse=False)

        self.week_emb = nn.Embedding(
                num_embeddings=self.config.total_weekdays, embedding_dim=self.args.week_dim, scale_grad_by_freq=False, sparse=False)


        self.G = nn.ModuleList([nn.Sequential(MLP(self.args.dim_c, self.args.dim_hidden, self.args.dim_h, self.args.num_hidden, activation), nn.Softplus()) for i in range(3)])

        self.L = nn.Sequential(MLP(3*(self.args.dim_c+self.args.dim_h)+self.args.hour_dim+self.args.week_dim, self.args.dim_hidden, len(self.config.P2id), self.args.num_hidden, activation=nn.CELU()), SoftPlus())

        self.W = MLP(self.args.dim_c+self.args.f_embedding_dim, self.args.dim_hidden, self.args.dim_h, self.args.num_hidden, activation)

        #self.evnt_embed = lambda k: (torch.arange(0, len(self.config.Pid2N)).to(device) == k).float()

        self.evnt_embed = nn.Embedding(
            num_embeddings=len(self.config.Pid2N), embedding_dim=self.args.f_embedding_dim)

        self.noise_linear = MLP(self.args.dim_c, self.args.dim_hidden, self.args.dim_c, 1, activation)


    def t2weekhour_emb(self,t):
        if (t>=0).all():
            #print('t to weekhour!')
            weekday = ((t*self.args.shrink/24).long()%7).unsqueeze(dim=0)
            hour = (t*self.args.shrink%24).long().unsqueeze(dim=0)
            weekday_emb = self.week_emb(weekday)
            hour_emb = self.hour_emb(hour)
            return torch.cat((weekday_emb,hour_emb),dim=-1)

        else:
            return torch.zeros([1,self.args.hour_dim+self.args.week_dim]).to(self.device)
        
    def forward(self, t, z): # z: batch_size * 3 * dim_z

        week_hour_emb = self.t2weekhour_emb(t).repeat(z.shape[0],1)

        #week_hour_emb = torch.zeros([1,self.args.hour_dim+self.args.week_dim]).repeat(z.shape[0],1).to(self.device)

        f = []

        for i in range(3):
            c = z[...,i*(self.args.dim_c+self.args.dim_h):i*(self.args.dim_c+self.args.dim_h)+self.args.dim_c]
            h = z[...,i*(self.args.dim_c+self.args.dim_h)+self.args.dim_c:(i+1)*(self.args.dim_c+self.args.dim_h)]

            c_h = z[...,i*(self.args.dim_c+self.args.dim_h):(i+1)*(self.args.dim_c+self.args.dim_h)]

            temp = torch.cat((c_h,week_hour_emb),dim=-1)

            dcdt = self.F[i](temp)

            if self.args.ortho==1:
                dcdt = dcdt - (dcdt*c).sum(dim=-1, keepdim=True) / (c*c).sum(dim=-1, keepdim=True) * c

            dhdt = -self.G[i](c)*h

            f.append(torch.cat((dcdt, dhdt), dim=-1))

        return torch.cat(f, dim=-1)  # the same shape as zs

    def next_read_jump(self,t0,t1):
        #assert self.jump_type == "read", "next_read_jump must be called with jump_type = read"
        assert t0 != t1, "t0 can not equal t1"

        t = t1
        inf = sys.maxsize
        if t0 < t1:  # forward
            idx = bisect.bisect_right(self.evnts, (t0, inf, inf, inf))
            if idx != len(self.evnts):
                t = min(t1, torch.tensor(self.evnts[idx][0], dtype=torch.float64).to(self.device))
        else:  # backward
            idx = bisect.bisect_left(self.evnts, (t0, -inf, -inf, -inf))
            if idx > 0:
                t = max(t1, torch.tensor(self.evnts[idx-1][0], dtype=torch.float64).to(self.device))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t, z):
        #assert self.jump_type == "read", "read_jump must be called with jump_type = read"
        dz = torch.zeros(z.shape).to(self.device)
        inf = sys.maxsize
        lid = bisect.bisect_left(self.evnts, (t, -inf, -inf, -inf))
        rid = bisect.bisect_right(self.evnts, (t, inf, inf, inf))

        for evnt in self.evnts[lid:rid]:
            # find location and type of event
            loc, k = evnt[1:-1], evnt[-1]
            n = self.config.Pid2N[k]
            c = z[...,n*(self.args.dim_c+self.args.dim_h):n*(self.args.dim_c+self.args.dim_h)+self.args.dim_c]
            # encode of event k
            kv = self.evnt_embed(torch.tensor(k).to(self.device))
            # add to jump
            dz[loc][n*(self.args.dim_c+self.args.dim_h)+self.args.dim_c:(n+1)*(self.args.dim_c+self.args.dim_h)] += self.W(torch.cat((c[loc], kv), dim=-1))
        return dz

    def next_simulated_jump(self, t0, z0, t1):

        week_hour_emb = self.t2weekhour_emb(t0).repeat(z0.shape[0],1)
        
        #week_hour_emb = torch.zeros([1,self.args.hour_dim+self.args.week_dim]).repeat(z0.shape[0],1).to(self.device)
        
        z0_hour_week = torch.cat((z0,week_hour_emb),dim=-1)

        m = torch.distributions.Exponential(self.L(z0_hour_week).double())
        # next arrival time
        tt = t0 + m.sample()
        tt_min = tt.min()

        if tt_min <= t1:
            dN = (tt == tt_min).float()
            next_t = tt_min
        else:
            dN = torch.zeros(tt.shape).to(self.device)
            next_t = t1

        return dN, next_t

    def simulated_jump(self, dN, t, z):
        #assert self.jump_type == "simulate", "simulate_jump must be called with jump_type = simulate"
        dz = torch.zeros(z.shape).to(self.device)
        sequence = []
        c = z[..., :self.args.dim_c]
        for idx in dN.nonzero():
            # find location and type of event
            loc, k = tuple(idx[:-1]), idx[-1]
            n = self.config.Pid2N[k.item()]

            c = z[...,n*(self.args.dim_c+self.args.dim_h):n*(self.args.dim_c+self.args.dim_h)+self.args.dim_c]
            
            kv = self.evnt_embed(torch.tensor(k).to(self.device))

            temp = (t.item(),) + (loc[0].item(),) + (k.item(),)

            sequence.extend([temp])

            # add to jump
            dz[loc][n*(self.args.dim_c+self.args.dim_h)+self.args.dim_c:(n+1)*(self.args.dim_c+self.args.dim_h)] += self.W(torch.cat((c[loc], kv), dim=-1))

        self.evnts.extend(sequence)

        return dz



class ODEJumpFunc_origin(nn.Module):

    def __init__(self, config, args, device, activation=nn.CELU()):
        super(ODEJumpFunc_origin, self).__init__()

        self.jump_type='simulate'
        self.config = config
        self.args = args
        self.device = device
        self.evnts = []

        self.F = MLP(self.args.dim_c+self.args.dim_h, self.args.dim_hidden,self.args.dim_c,self.args.num_hidden, activation)

        self.G = nn.Sequential(MLP(self.args.dim_c, self.args.dim_hidden, self.args.dim_h, self.args.num_hidden, activation), nn.Softplus())

        self.L = nn.Sequential(MLP(self.args.dim_c+self.args.dim_h, self.args.dim_hidden, len(self.config.P2id), self.args.num_hidden, activation=nn.CELU()), SoftPlus())

        self.W = MLP(self.args.dim_c+self.args.f_embedding_dim, self.args.dim_hidden, self.args.dim_h, self.args.num_hidden, activation)

        self.evnt_embed = nn.Embedding(
            num_embeddings=len(self.config.Pid2N), embedding_dim=self.args.f_embedding_dim)

        self.noise_linear = MLP(self.args.dim_c, self.args.dim_hidden, self.args.dim_c, 1, activation)

        

    def forward(self, t, z): # z: batch_size  dim_z

        c = z[..., :self.args.dim_c]
        h = z[..., self.args.dim_c:]

        dcdt = self.F(z)

        # orthogonalize dc w.r.t. to c
        if self.args.ortho:
            dcdt = dcdt - (dcdt*c).sum(dim=-1, keepdim=True) / (c*c).sum(dim=-1, keepdim=True) * c

        dhdt = -self.G(c) * h


        return torch.cat((dcdt, dhdt), dim=-1)


    def next_read_jump(self,t0,t1):
        assert self.jump_type == "read", "next_read_jump must be called with jump_type = read"
        assert t0 != t1, "t0 can not equal t1"

        t = t1
        inf = sys.maxsize
        if t0 < t1:  # forward
            idx = bisect.bisect_right(self.evnts, (t0, inf, inf, inf))
            if idx != len(self.evnts):
                t = min(t1, torch.tensor(self.evnts[idx][0], dtype=torch.float64).to(self.device))
        else:  # backward
            idx = bisect.bisect_left(self.evnts, (t0, -inf, -inf, -inf))
            if idx > 0:
                t = max(t1, torch.tensor(self.evnts[idx-1][0], dtype=torch.float64).to(self.device))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t, z):
        assert self.jump_type == "read", "read_jump must be called with jump_type = read"
        dz = torch.zeros(z.shape).to(self.device)

        inf = sys.maxsize
        lid = bisect.bisect_left(self.evnts, (t, -inf, -inf, -inf))
        rid = bisect.bisect_right(self.evnts, (t, inf, inf, inf))

        c = z[..., :self.args.dim_c]
        for evnt in self.evnts[lid:rid]:
            # find location and type of event
            loc, k = evnt[1:-1], evnt[-1]

            # encode of event k
            kv = self.evnt_embed(torch.tensor(k).to(self.device))

            # add to jump
            dz[loc][self.args.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1))

        return dz

    def next_simulated_jump(self, t0, z0, t1):

        m = torch.distributions.Exponential(self.L(z0).double())
        # next arrival time
        tt = t0 + m.sample() 
        tt_min = tt.min()

        print('t_min',tt_min)

        if tt_min <= t1:
            dN = (tt == tt_min).float()
        else:
            dN = torch.zeros(tt.shape).to(self.device)

        next_t = min(tt_min, t1)

        return dN, next_t

    def simulated_jump(self, dN, t, z):
        assert self.jump_type == "simulate", "simulate_jump must be called with jump_type = simulate"
        dz = torch.zeros(z.shape).to(self.device)
        sequence = []
        c = z[..., :self.args.dim_c]

        
        for idx in dN.nonzero():
            # find location and type of event
            
            loc, k = tuple(idx[:-1]), idx[-1]
            ne = int(dN[tuple(idx)])

            for _ in range(ne):
                # encode of event k
                kv = self.evnt_embed(torch.tensor(k).to(self.device))
                sequence.extend([(t.item(),) + (loc[0].item(),) + (k.item(),)])

                dz[loc][self.args.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1)).squeeze(dim=0)

        self.evnts.extend(sequence)
        return dz
