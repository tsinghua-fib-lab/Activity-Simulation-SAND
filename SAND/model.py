import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import math
from torch.autograd import Variable

from utils import *


class Policy_net(nn.Module):
    def __init__(self,config,args,device):
        super(Policy_net, self).__init__()
        self.device = device
        self.config = config
        self.args = args
        self.func = ODEJumpFunc(self.config, self.args, self.device, activation=nn.CELU()).to(self.device)

    def get_z(self,evnts,z0):

        tgrid = np.round(np.arange(0,self.config.traj_length,self.config.traj_length-1/self.args.shrink),decimals=self.args.decimals)

        tevnt = np.array([evnt[0] for evnt in evnts])

        tevolve = np.sort(np.unique(np.concatenate((tgrid,tevnt))))

        t2tid = {t: tid for tid, t in enumerate(tevolve)}

        tevolve = torch.tensor(tevolve,dtype=torch.float64).to(self.device)

        self.func.jump_type = 'read'

        tse = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]

        self.func.evnts = evnts

        z = odeint(self.func, z0, tevolve, method = self.args.ODE_method, rtol = self.args.rtol, atol = self.args.atol)

        return z, tse, tevolve

    def evaluate_actions(self, evnts_list, z0):

        z, tse,tevolve = self.get_z(evnts_list,z0)

        weekday = (tevolve*self.args.shrink/24).long()%7
        hour = (tevolve*self.args.shrink%24).long()

        weekday_emb = self.func.week_emb(weekday)
        hour_emb = self.func.hour_emb(hour)

        week_hour_emb = torch.cat((weekday_emb,hour_emb),dim=-1).unsqueeze(dim=1).repeat(1,z.shape[1],1)

        z_week_hour = torch.cat((z ,week_hour_emb),dim=-1)

        action_log_prob,action_dist_entropy = [],[]

        Lambda = self.func.L(z_week_hour)

        for evnt in tse:
            param = Lambda[evnt[:2]]
            param = torch.distributions.Categorical(param.float())
            k = torch.tensor(evnt[2]).to(self.device)
            action_log_prob.append(param.log_prob(k).unsqueeze(dim=0))
            action_dist_entropy.append(param.entropy().unsqueeze(dim=0))

        action_log_prob = torch.cat(action_log_prob,dim=0)
        action_dist_entropy = torch.cat(action_dist_entropy,dim=0)

        return action_log_prob, action_dist_entropy
    
    def forward(self,z0,tevolve):
        z = odeint(self.func, z0, tevolve,method=self.args.ODE_method, rtol=self.args.rtol, atol=self.args.atol)
        return z

    def act(self,z0,tevolve):
        z = self.forward(z0,tevolve)
        return z.cpu().detach()

class Value_net(nn.Module):
    def __init__(self,config,args,device):
        super(Value_net, self).__init__()
        self.config = config
        self.args = args
        self.device = device

        self.critic_linear = nn.Linear(self.args.dim_hidden, 1)

        self.critic = nn.Sequential(MLP(3*self.args.f_embedding_dim+1*(self.args.week_dim+self.args.hour_dim), self.args.dim_hidden, self.args.dim_hidden, self.args.num_hidden, activation=nn.Tanh()), nn.Tanh())

        # embedding layer
        self.poi_embedding = nn.Embedding(
            num_embeddings=len(self.config.Pid2N), embedding_dim=self.args.f_embedding_dim)
        self.need_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.args.f_embedding_dim)
        self.time_embedding = nn.Parameter(torch.randn([1,self.args.f_embedding_dim]).float())

        self.poi_proj = nn.Linear(self.args.f_embedding_dim, self.args.f_embedding_dim)
        self.need_proj = nn.Linear(self.args.f_embedding_dim, self.args.f_embedding_dim)
        self.interval_proj = nn.Linear(self.args.f_embedding_dim, self.args.f_embedding_dim)
        self.week_proj = nn.Linear(self.args.week_dim, self.args.f_embedding_dim)
        self.hour_proj = nn.Linear(self.args.hour_dim, self.args.f_embedding_dim)

        self.attn_vector_poi = nn.Parameter(torch.randn([self.args.f_embedding_dim,1]).float())
        self.attn_vector_need = nn.Parameter(torch.randn([self.args.f_embedding_dim,1]).float())
        self.attn_vector_interval = nn.Parameter(torch.randn([self.args.f_embedding_dim,1]).float())
        self.attn_vector_week = nn.Parameter(torch.randn([self.args.f_embedding_dim,1]).float())
        self.attn_vector_hour = nn.Parameter(torch.randn([self.args.f_embedding_dim,1]).float())

        self.hour_emb = nn.Embedding(
                num_embeddings=self.config.total_hours, embedding_dim=self.args.hour_dim, scale_grad_by_freq=False, sparse=False)

        self.week_emb = nn.Embedding(
                num_embeddings=self.config.total_weekdays, embedding_dim=self.args.week_dim, scale_grad_by_freq=False, sparse=False)

    def forward(self, poi_list, need_list, interval_list, weekday_list, hour_list,mask):
    #poi_state,need_state,interval_state,interval,k,mask):

        #sequence features
        need_emb = self.need_embedding(need_list)
        poi_emb = self.poi_embedding(poi_list)
        interval_emb = interval_list * self.time_embedding
        weekday_list = self.week_emb(weekday_list)
        hour_list= self.hour_emb(hour_list)

        # attention
        H_need = torch.tanh(self.need_proj(need_emb))
        w_need = H_need.matmul(self.attn_vector_need)
        w_need.masked_fill_(mask,-1e9)
        w_score_need = F.softmax(w_need,dim=1)
        need_out = torch.sum(need_emb.mul(w_score_need),dim=1)

        H_poi = torch.tanh(self.poi_proj(poi_emb))
        w_poi = H_poi.matmul(self.attn_vector_poi)
        w_poi.masked_fill_(mask,-1e9)
        w_score_poi = F.softmax(w_poi,dim=1)
        poi_out = torch.sum(poi_emb.mul(w_score_poi),dim=1)

        H_interval = torch.tanh(self.interval_proj(interval_emb))
        w_interval = H_interval.matmul(self.attn_vector_interval)
        w_interval.masked_fill_(mask,-1e9)
        w_score_interval = F.softmax(w_interval,dim=1)
        interval_out = torch.sum(interval_emb.mul(w_score_interval),dim=1)

        H_week = torch.tanh(self.week_proj(weekday_list))
        w_week = H_week.matmul(self.attn_vector_week)
        w_week.masked_fill_(mask,-1e9)
        w_score_week = F.softmax(w_week,dim=1)
        week_out = torch.sum(weekday_list.mul(w_score_week),dim=1)

        H_hour = torch.tanh(self.hour_proj(hour_list))
        w_hour = H_hour.matmul(self.attn_vector_hour)
        w_hour.masked_fill_(mask,-1e9)
        w_score_hour = F.softmax(w_hour,dim=1)
        hour_out = torch.sum(hour_list.mul(w_score_hour),dim=1)

        x = torch.cat((need_out, poi_out, interval_out, week_out, hour_out),dim=-1)

        value = self.critic_linear(self.critic(x)).squeeze(dim=1)

        return value


class Discriminator(nn.Module):
    def __init__(self,config,args,device):
        super(Discriminator, self).__init__()
        self.config = config
        self.args = args
        self.device = device

        self.mlp_layer = nn.Sequential(FC(6*self.args.d_embedding_dim+2*(self.args.hour_dim+self.args.week_dim),1,use_relu=False),nn.Sigmoid())

        # embedding layer
        self.poi_embedding = nn.Embedding(
            num_embeddings=len(self.config.Pid2N), embedding_dim=self.args.d_embedding_dim)
        self.need_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.args.d_embedding_dim)
        self.time_embedding = nn.Parameter(torch.randn([1,self.args.d_embedding_dim]).float())

        self.poi_proj = nn.Linear(self.args.d_embedding_dim, self.args.d_embedding_dim)
        self.need_proj = nn.Linear(self.args.d_embedding_dim, self.args.d_embedding_dim)
        self.interval_proj = nn.Linear(self.args.d_embedding_dim, self.args.d_embedding_dim)
        self.week_proj = nn.Linear(self.args.week_dim, self.args.d_embedding_dim)
        self.hour_proj = nn.Linear(self.args.hour_dim, self.args.d_embedding_dim)

        self.attn_vector_poi = nn.Parameter(torch.randn([self.args.d_embedding_dim,1]).float())
        self.attn_vector_need = nn.Parameter(torch.randn([self.args.d_embedding_dim,1]).float())
        self.attn_vector_interval = nn.Parameter(torch.randn([self.args.d_embedding_dim,1]).float())
        self.attn_vector_week = nn.Parameter(torch.randn([self.args.d_embedding_dim,1]).float())
        self.attn_vector_hour = nn.Parameter(torch.randn([self.args.d_embedding_dim,1]).float())

        self.hour_emb = nn.Embedding(
                num_embeddings=self.config.total_hours, embedding_dim=self.args.hour_dim, scale_grad_by_freq=False, sparse=False)

        self.week_emb = nn.Embedding(
                num_embeddings=self.config.total_weekdays, embedding_dim=self.args.week_dim, scale_grad_by_freq=False, sparse=False)
    
    def forward(self, poi_list, need_list, interval_list, weekday_list, hour_list, interval,k,need,weekday,hour,mask):
    #poi_state,need_state,interval_state,interval,k,mask):

        #sequence features
        need_emb = self.need_embedding(need_list)
        poi_emb = self.poi_embedding(poi_list)
        interval_emb = interval_list * self.time_embedding
        weekday_list = self.week_emb(weekday_list)
        hour_list= self.hour_emb(hour_list)

        # attention
        H_need = torch.tanh(self.need_proj(need_emb))
        w_need = H_need.matmul(self.attn_vector_need)
        w_need.masked_fill_(mask,-1e9)
        w_score_need = F.softmax(w_need,dim=1)
        need_out = torch.sum(need_emb.mul(w_score_need),dim=1)

        H_poi = torch.tanh(self.poi_proj(poi_emb))
        w_poi = H_poi.matmul(self.attn_vector_poi)
        w_poi.masked_fill_(mask,-1e9)
        w_score_poi = F.softmax(w_poi,dim=1)
        poi_out = torch.sum(poi_emb.mul(w_score_poi),dim=1)

        H_interval = torch.tanh(self.interval_proj(interval_emb))
        w_interval = H_interval.matmul(self.attn_vector_interval)
        w_interval.masked_fill_(mask,-1e9)
        w_score_interval = F.softmax(w_interval,dim=1)
        interval_out = torch.sum(interval_emb.mul(w_score_interval),dim=1)

        H_week = torch.tanh(self.week_proj(weekday_list))
        w_week = H_week.matmul(self.attn_vector_week)
        w_week.masked_fill_(mask,-1e9)
        w_score_week = F.softmax(w_week,dim=1)
        week_out = torch.sum(weekday_list.mul(w_score_week),dim=1)

        H_hour = torch.tanh(self.hour_proj(hour_list))
        w_hour = H_hour.matmul(self.attn_vector_hour)
        w_hour.masked_fill_(mask,-1e9)
        w_score_hour = F.softmax(w_hour,dim=1)
        hour_out = torch.sum(hour_list.mul(w_score_hour),dim=1)

        act_emb = self.poi_embedding(k)
        need_act_emb = self.need_embedding(need)
        time_emb = interval*self.time_embedding
        weekday_emb = self.week_emb(weekday)
        hour_emb = self.hour_emb(hour)

        x = torch.cat((need_out, poi_out, interval_out, week_out, hour_out, act_emb,time_emb,need_act_emb,weekday_emb,hour_emb),dim=-1)

        reward = self.mlp_layer(x)

        return reward

