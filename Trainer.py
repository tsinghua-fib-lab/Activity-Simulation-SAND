from torch.autograd.grad_mode import no_grad
from model import *
from Storage import *
from utils import *
from evaluation import Evaluation
from torchdiffeq import odeint_adjoint as odeint
from sklearn.metrics import accuracy_score
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import copy
import time
import setproctitle
import random
import os
import json
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

def LR_warmup(lr,epoch_num, epoch_current):
    return lr * epoch_current / epoch_num


class Trainer(object):
    def __init__(self, config, args, real_file, test_file, fake_file, device):
        self.config = config
        self.args = args
        self.device = device

        self.real_file = real_file
        self.test_file = test_file
        self.fake_file = fake_file

        self.policy_net = Policy_net(config,args,self.device).to(self.device)
        self.discriminator = Discriminator(config,args,self.device).to(self.device)
        self.value_net = Value_net(config,args,self.device).to(self.device)

        self.buffer = RolloutStorage(config, args)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.policy_lr, weight_decay=args.weight_decay)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr = args.dis_lr_pretrain,weight_decay = args.weight_decay)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),lr = args.value_lr,weight_decay = args.weight_decay)

        self.value_scheduler = ReduceLROnPlateau(self.value_optimizer, 'min',min_lr=1e-4, patience=5,factor=0.5)

        # loss
        self.dis_loss = nn.BCELoss(reduction='sum').to(self.device)

        # evaluation
        self.evaluation = Evaluation(config,args)

    def ppo_train(self):
        print('ppo start!')

        t = 0

        buffer = []

        while t < len(self.buffer):
            temp = []
            while True:
                temp.append(self.buffer.memory[t])
                if self.buffer.memory[t][3]:
                    t += 1
                    break
                t += 1
            buffer.append(temp)

        _, _, _, _, _, action_log_probs, _, _, returns,advantages = zip(*self.buffer.memory)

        old_action_probs = torch.FloatTensor(action_log_probs).detach().to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(dim=1).detach().to(self.device)
        advantages = torch.FloatTensor(advantages)

        if self.args.adv_norm==1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        advantages = advantages.detach().to(self.device)

        pos = ((advantages>0).float().sum()/advantages.shape[0]).item()

        print('pos-percentage: ',pos)


        buffer_evnts = []

        for b_index, b in enumerate(buffer):
            _, action, _, _, _, _,_,z0,_,_ = zip(*b)
            buffer_evnts += [(a[-1],b_index,a[1]) for a in action]

        time_temp = [i[0] for i in buffer_evnts]
        sort_id = sorted(range(len(time_temp)),key=lambda k:time_temp[k])

        old_action_probs = old_action_probs[sort_id]
        advantages = advantages[sort_id]

        buffer_evnts.sort()
        
        for _ in range(self.args.value_iter):
            buffer_value = []
            for b_index, b in enumerate(buffer):
                state, _, _, _, _, _,_,_,_,_ = zip(*b)
                for st in state:
                    a1, a2, a3, a4, a5, _, _, _, _, _, a6 = self.state_action_tensor(st,None)
                    buffer_value.append(self.value_net(a1.to(self.device), a2.to(self.device), a3.to(self.device), a4.to(self.device), a5.to(self.device), a6.to(self.device)))

            values = torch.cat(buffer_value,dim=0).to(self.device)
            value_loss = 0.5 * (returns-values.unsqueeze(dim=1)).pow(2).mean()

            self.value_optimizer.zero_grad()

            value_loss.backward()

            self.value_optimizer.step()


        start = time.time()

        for _ in range(self.args.ppo_epoch):

            s = time.time()
            buffer_z0 = []

            for b_index, b in enumerate(buffer):
                _, action, _, _, _, _,_,z0,_,_ = zip(*b)
                c0_origin,h0=z0[-1]
                c0 = [self.policy_net.func.noise_linear(i.to(self.device)) for i in c0_origin]
                z0_s = torch.cat([torch.cat((c0[i],h0[i].to(self.device)),dim=-1) for i in range(3)],dim=-1).unsqueeze(dim=0)
        

                buffer_z0.append(z0_s)

            buffer_z0 = torch.cat(buffer_z0,dim=0)

            action_log_probs, action_dist_entropys = self.policy_net.evaluate_actions(buffer_evnts,buffer_z0)

            print('calculate z: {}'.format((time.time()-s)/60.0))

            s = time.time()

            print('action_log',action_log_probs.shape, old_action_probs.shape)
            
            ratio = torch.exp(action_log_probs-old_action_probs)

            surr1 = 0.5 * ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.args.clip_param, 1. + self.args.clip_param,) * advantages

            action_loss = -torch.min(surr1, surr2).mean()

            loss = action_loss - action_dist_entropys.mean() * self.args.entropy_coef

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),self.args.clip_grad)
            print('backward: {}'.format((time.time()-s)/60.0))
            self.policy_optimizer.step()
            print('ppo loss: {}'.format(loss.item()))
            entropy_loss = action_dist_entropys.mean()
            print('entropy loss: {}'.format(entropy_loss.item()))

            if self.args.coef_decay_epoch==1:
                self.args.entropy_coef *= self.args.entropy_decay

        if self.args.coef_decay_epoch==0:
            self.args.entropy_coef *= self.args.entropy_decay

        print('time of ppo training: {}'.format((time.time()-start)/60.0))
        return 0


    def t2weekhour(self,t):
        weekday = int(t*self.args.shrink/24)%7
        hour = int(t*self.args.shrink%24)

        return weekday,hour

    def t2weekhour_batch(self,tevolve,batch):
        weekday = (tevolve*self.args.shrink/24).long()%7
        hour = (tevolve*self.args.shrink%24).long()

        weekday_emb = self.policy_net.func.week_emb(weekday)
        hour_emb = self.policy_net.func.hour_emb(hour)

        week_hour_emb = torch.cat((weekday_emb,hour_emb),dim=-1).unsqueeze(dim=1).repeat(1,batch,1)

        return week_hour_emb

    def sample_real_data(self, batch_seq,f):

        
        file = random.sample(f,batch_seq)

        state_action = []

        for u in file:
            for index, i in enumerate(u):
                if index==0:
                    continue
                s = [[t[0],t[1],self.config.Pid2N[t[1]],self.t2weekhour(t[0])[0],self.t2weekhour(t[0])[1]] for t in u[:index]] # past evnts set
                s = [[t[0] if tt==0 else t[0]-s[tt-1][0],t[1],t[2],t[3],t[4]] for tt, t in enumerate(s)]
                week,hour = self.t2weekhour(i[0])
                a = (i[0]-u[index-1][0],i[1],self.config.Pid2N[i[1]],week,hour) #  (interval,poi,need)
                state_action.append((s,)+a)
        state_action = random.sample(state_action, len(state_action))
        return state_action

    def discriminator_train(self,Type='generate'):

        # generate

        if Type == 'generate':

            gene_state, gene_action =  zip(*self.buffer.dis_memory)

            gene_evnts_list = [[[i[0],i[1],self.config.Pid2N[i[1]],i[2],i[3]] for i in seq] for seq in gene_state] # interval, k , need, weekday, hour
            gene_interval, gene_k, gene_need, gene_weekday, gene_hour, _ = zip(*gene_action)

        else:

            batch_seq = 1000 if len(self.fake_file)>1000 else len(self.fake_file)
            gene_state_action = self.sample_real_data(batch_seq,self.fake_file)

            gene_evnts_list, gene_interval,gene_k,gene_need,gene_weekday,gene_hour = zip(*gene_state_action)

            gene_evnts_list = list(gene_evnts_list)
    

        gene_k = torch.LongTensor(gene_k)
        gene_need = torch.LongTensor(gene_need)
        gene_interval = torch.FloatTensor(gene_interval)
        gene_weekday = torch.LongTensor(gene_weekday)
        gene_hour = torch.LongTensor(gene_hour)

        gene_labels = torch.FloatTensor(gene_interval.shape[0],1).fill_(0)

        # expert

        batch_seq = round(gene_labels.shape[0]/self.config.real_len_avg)

        batch_seq = batch_seq if batch_seq < len(self.real_file) else len(self.real_file)
        expert_state_action = self.sample_real_data(batch_seq,self.real_file)

        expert_evnts_list, interval,k,need,weekday,hour = zip(*expert_state_action)

        expert_evnts_list = list(expert_evnts_list)

        expert_interval = torch.FloatTensor(interval)
        expert_k = torch.LongTensor(k)
        expert_need = torch.LongTensor(need)
        expert_week = torch.LongTensor(weekday)
        expert_hour = torch.LongTensor(hour)

        expert_labels = torch.FloatTensor(expert_interval.shape[0],1).fill_(1)

        evnts_list_all = [torch.tensor(i) for i in expert_evnts_list + gene_evnts_list]
        seq_len = [len(i) for i in evnts_list_all]


        evnts_list_all_pad = rnn_utils.pad_sequence(evnts_list_all,padding_value=0).permute(1,0,2)
        mask = torch.ones(evnts_list_all_pad.shape[0],evnts_list_all_pad.shape[1])
        for index,l in enumerate(seq_len):
            mask[index][:l] = 0
        mask = mask.bool().unsqueeze(dim=2)

        list_poi = evnts_list_all_pad[...,1].long()
        list_need = evnts_list_all_pad[...,2].long()
        list_interval = evnts_list_all_pad[...,0].unsqueeze(dim=2).float()

        list_weekday = evnts_list_all_pad[...,3].long()
        list_hour= evnts_list_all_pad[...,4].long()

        k_all = torch.cat((expert_k,gene_k),dim=0)
        interval_all = torch.cat((expert_interval,gene_interval),dim=0).unsqueeze(dim=1)

        need_all = torch.cat((expert_need,gene_need),dim=0)

        weekday_all = torch.cat((expert_week,gene_weekday),dim=0)
        hour_all = torch.cat((expert_hour,gene_hour),dim=0)

        labels_all = torch.cat((expert_labels,gene_labels),dim=0)

        target = np.round(labels_all.cpu().detach())


        for dis_id in range(self.args.dis_iter):

            pred = []

            loss_all = 0.0

            for batch_id in range(int(target.shape[0]/self.args.dis_batch)+1):

                reward = self.discriminator(list_poi[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),list_need[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),list_interval[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),list_weekday[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device), list_hour[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),interval_all[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),k_all[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),need_all[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),weekday_all[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),hour_all[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device),mask[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device))

                dis_loss = self.dis_loss(reward, labels_all[batch_id*self.args.dis_batch:(batch_id+1)*self.args.dis_batch].to(self.device))

                self.discriminator_optimizer.zero_grad()

                pred += np.round(reward.cpu().detach()).tolist()

                dis_loss.backward()

                self.discriminator_optimizer.step()

                loss_all += dis_loss.item()
                
            acc = accuracy_score(pred,target)

        return 0

    def state_action_tensor(self,state,action):

        if action is not None:
            interval, k, need, weekday, hour, _ = action
            k = torch.LongTensor([k])
            need = torch.LongTensor([need])
            interval = torch.FloatTensor([interval])

            weekday_emb = torch.LongTensor([weekday])
            hour_emb = torch.LongTensor([hour])

        else:
            interval = None
            k = None
            need = None
            interval = None
            weekday_emb = None
            hour_emb = None

        evnts_list = state

        poi_list = torch.LongTensor([i[1] for i in evnts_list]).unsqueeze(dim=0)
        need_list = torch.LongTensor([i[2] for i in evnts_list]).unsqueeze(dim=0)
        interval_list = torch.FloatTensor([i[0] for i in evnts_list]).unsqueeze(dim=0).unsqueeze(dim=2)

        weekday_list = torch.LongTensor([i[3] for i in evnts_list]).unsqueeze(dim=0)

        hour_list = torch.LongTensor([i[4] for i in evnts_list]).unsqueeze(dim=0)


        if poi_list.shape[0]==1:
            mask = torch.zeros(poi_list.shape[0],poi_list.shape[1]).unsqueeze(dim=2).bool()
        else:
            seq_len = [len(i) for i in evnts_list]

            evnts_list_pad = rnn_utils.pad_sequence(evnts_list,padding_value=0).permute(1,0,2)
            mask = torch.ones(evnts_list_pad.shape[0],evnts_list_pad.shape[1])
            for index,l in enumerate(seq_len):
                mask[index][:l] = 0
            mask = mask.bool().unsqueeze(dim=2)

        return poi_list,need_list,interval_list, weekday_list,hour_list,interval, k, need, weekday_emb, hour_emb,mask


    def get_reward(self, state, action):

        poi_list,need_list,interval_list, weekday_list,hour_list,interval, k, need, weekday_emb, hour_emb,mask = self.state_action_tensor(state,action)

        d_reward = self.discriminator(poi_list.to(self.device),need_list.to(self.device),interval_list.to(self.device), weekday_list.to(self.device),hour_list.to(self.device),interval.to(self.device), k.to(self.device), need.to(self.device), weekday_emb.to(self.device), hour_emb.to(self.device),mask.to(self.device))
        log_reward = d_reward.log()

        if log_reward.shape[0]==1:
            return log_reward.detach().item() # 这个detach很关键，防止影响后面的discriminator训练
        else:
            return log_reward.cpu().detach().sum()

    def JSD_cal(self,real,fake):

        poi_distinct = self.evaluation.get_JSD(real,fake,JSD_type='0')
        event_cnt = self.evaluation.get_JSD(real,fake,JSD_type='1')
        interval = self.evaluation.get_JSD(real,fake,JSD_type='2')
        poi_cate = self.evaluation.get_JSD(real,fake,JSD_type='3')
        need = self.evaluation.get_JSD(real,fake,JSD_type='4')
        week = self.evaluation.get_JSD(real,fake,JSD_type='5')
        hour = self.evaluation.get_JSD(real,fake,JSD_type='6')
        poi_cnt = self.evaluation.get_JSD(real,fake,JSD_type='7')
        diversity = self.evaluation.get_JSD(real,fake,JSD_type='8')
        poi_interval = self.evaluation.get_JSD(real,fake,JSD_type='9')
        need_interval = self.evaluation.get_JSD(real,fake,JSD_type='10')

        return poi_distinct, event_cnt, interval, poi_cate, need, week, hour,poi_cnt,diversity, poi_interval,need_interval


    def generate_final_data(self,generate_num=2000):

        print('generate!')

        tgrid = np.arange(0,self.config.traj_length,self.config.traj_length-1/self.args.shrink)

        tgrid = torch.FloatTensor(tgrid)

        generate_data = []

        for _ in range(int(generate_num/self.args.test_batch)):

            with torch.no_grad():

                c0 = [self.policy_net.func.noise_linear(torch.randn(self.args.test_batch,self.args.dim_c).to(self.device)) for i in range(3)]

                h0 = [torch.zeros(self.args.test_batch,self.args.dim_h).to(self.device) for i in range(3)]

                # z0:
                z0 = torch.cat([torch.cat((c0[i],h0[i]),dim=-1) for i in range(3)],dim=-1)

                self.policy_net.func.evnts = []
                self.policy_net.func.jump_type = 'simulate'

                _ = self.policy_net.act(z0,tgrid.to(self.device))

                print('evnts num: {}'.format(len(self.policy_net.func.evnts)))

                evnts_list = [[] for _ in range(self.args.test_batch)]

                self.policy_net.func.evnts.sort(key=lambda x:(x[1],x[0]))

                for evnt in self.policy_net.func.evnts:
                    week,hour = self.t2weekhour(evnt[0])
                    evnts_list[int(evnt[1])].append([evnt[0],evnt[2],self.config.Pid2N[evnt[2]],week,hour])

                for index in range(self.args.test_batch):
                    evnts_list[index].sort()
                    generate_data.append(evnts_list[index])

        with open(self.args.generate_final_path+'gene_data.json', 'w') as f:
            json.dump(generate_data,f)

    
    def evaluation_JSD(self,epoch = 0):

        tgrid = np.arange(0,self.config.traj_length,self.config.traj_length-1/self.args.shrink)

        tgrid = torch.FloatTensor(tgrid)

        s = time.time()
        fake = []

        for _ in range(int(self.args.generate_cnt/self.args.test_batch)):

            with torch.no_grad():

                c0 = [self.policy_net.func.noise_linear(torch.randn(self.args.test_batch,self.args.dim_c).to(self.device)) for i in range(3)]

                h0 = [torch.zeros(self.args.test_batch,self.args.dim_h).to(self.device) for i in range(3)]

                # z0
                z0 = torch.cat([torch.cat((c0[i],h0[i]),dim=-1) for i in range(3)],dim=-1)

                self.policy_net.func.evnts = []
                self.policy_net.func.jump_type = 'simulate'

                _ = self.policy_net.act(z0,tgrid.to(self.device))

                print('evnts num: {}'.format(len(self.policy_net.func.evnts)))

                evnts_list = [[] for _ in range(self.args.test_batch)]

                self.policy_net.func.evnts.sort(key=lambda x:(x[1],x[0]))

                for evnt in self.policy_net.func.evnts:
                    week,hour = self.t2weekhour(evnt[0])
                    evnts_list[int(evnt[1])].append([evnt[0],evnt[2],self.config.Pid2N[evnt[2]],week,hour])

                for index in range(self.args.test_batch):
                    evnts_list[index].sort()
                    fake.append([[i[0],i[1],i[2],i[3],i[4],i[0]] for ii, i in enumerate(evnts_list[index])]) # time, poi, need, week, hour, time

                print('generate fake data: {} min'.format(round((time.time()-s)/60.0)))

        return self.JSD_cal(self.test_file,fake)

    def ODE_pretrain(self):

        self.policy_net.func.jump_type = 'read'

        pretrain_file = copy.deepcopy(self.real_file[:self.args.ODE_pretrain_expert])

        tc = lambda t: t

        tgrid = np.round(np.arange(0,self.config.traj_length,10/self.args.shrink),decimals = self.args.decimals)

        loss_all_min = 1e9
        s = 0

        self.policy_optimizer.param_groups[0]['lr'] = self.args.pretrain_lr1

        self.args.ODE_pretrain_batch = self.args.ODE_pretrain_batch if self.args.ODE_pretrain_batch < len(pretrain_file) else len(pretrain_file)

        print(len(pretrain_file),self.args.ODE_pretrain_batch)

        for epoch_id in range(self.args.ODE_pretrain_epoch):

            if epoch_id<=3:
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = LR_warmup(1e-2, 3, epoch_id)

            elif epoch_id <= 10 and epoch_id > 3:
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = 1e-2 - (epoch_id-3) * ((1e-2)-(1e-3))/(10-3)
            
            elif epoch_id <= 50 and epoch_id >10:
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = 1e-3 - (epoch_id-10) * ((1e-3)-(1e-4))/(30-10)

            loss_all = 0.0
            integrate_all = 0.0
            log_all = 0.0

            for batch_id in range(int(np.ceil(len(pretrain_file)/self.args.ODE_pretrain_batch))):

                s_all = time.time()

                print('batch_id: {}'.format(batch_id+1))

                file = pretrain_file[batch_id*self.args.ODE_pretrain_batch:(batch_id+1)*self.args.ODE_pretrain_batch]

                c0 = [self.policy_net.func.noise_linear(torch.randn(len(file),self.args.dim_c).to(self.device)) for i in range(3)]

                h0 = [torch.zeros(len(file),self.args.dim_h).to(self.device) for i in range(3)]

                # z0: the concatenation of three need levels
                z0 = torch.cat([torch.cat((c0[i],h0[i]),dim=-1) for i in range(3)],dim=-1)

                evnts = sorted([(tc(i[0]),seq_id,i[1]) for seq_id, seq in enumerate(file) for i in seq])

                tevnt = np.array([evnt[0] for evnt in evnts])

                tsave = np.sort(np.unique(np.concatenate((tgrid,tevnt))))

                t2tid = {t: tid for tid, t in enumerate(tsave)}

                tse = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]

                tsave = torch.FloatTensor(tsave).to(self.device)

                self.policy_net.func.evnts = evnts

                self.policy_net.func.jump_type='read'

                self.policy_optimizer.zero_grad()

                z = odeint(self.policy_net.func, z0, tsave, method=self.args.ODE_method, rtol=self.args.rtol, atol=self.args.atol).float()

                weekday = (tsave*self.args.shrink/24).long()%7
                hour = (tsave*self.args.shrink%24).long()
                weekday_emb = self.policy_net.func.week_emb(weekday)
                hour_emb = self.policy_net.func.hour_emb(hour)
                week_hour_emb = torch.cat((weekday_emb,hour_emb),dim=-1).unsqueeze(dim=1).repeat(1,z.shape[1],1)

                z_week_hour = torch.cat((z,week_hour_emb),dim=-1)
                
                lmbda = self.policy_net.func.L(z_week_hour)

                def integrate(tt, ll):
                    lm = (ll[:-1, ...] + ll[1:, ...]) / 2.0
                    dts = (tt[1:] - tt[:-1]).reshape((-1,)+(1,)*(len(lm.shape)-1)).float()
                    return (lm * dts).sum()

                integrate_loss = integrate(tsave, torch.sum(lmbda,dim=-1)) # 积分的一项

                log_likelihood = 0

                for evnt in tse:
                    log_likelihood -= torch.log(lmbda[evnt])

                loss = integrate_loss + log_likelihood

                start = time.time()
                
                loss.backward()
                self.policy_optimizer.step()

                print('total time: {}'.format((time.time()-s_all)/60.0))

                loss_all += loss.item()
                integrate_all += integrate_loss.item()
                log_all += log_likelihood.item()

            torch.cuda.empty_cache()



    def test_generate(self):

        setproctitle.setproctitle("generate")

        self.policy_net.load_state_dict(torch.load(self.args.trained_model,map_location='cuda:0'))

        print('load model finished!')

        self.generate_final_data(self.args.generate_num_down)

        return 0


    
    def overall_train(self):

        TIME = int(time.time())
        TIME = time.localtime(TIME)
        TIME = time.strftime("%Y-%m-%d %H:%M:%S",TIME)
        self.model_path = './model/{}/'.format(TIME)

        setproctitle.setproctitle("run_SAND")
        
        if not os.path.exists('./model/'):
            os.mkdir('./model/')

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        with open(self.model_path+'test_data.json','w') as f:
            json.dump(self.test_file,f)

        poi_distinct_max, event_cnt_max, interval_max, poi_cate_max, need_max, week_max, hour_max, poi_cnt_max,diversity_max = 1e9,1e9,1e9,1e9,1e9,1e9,1e9,1e9,1e9

        if self.args.is_pretrain==1:
            print('pretrain!')
            self.ODE_pretrain()

            #discriminator pretrain
            self.discriminator_train(Type='pretrain')
            self.args.dis_iter = 5
            self.discriminator_optimizer.param_groups[0]['lr'] = self.args.dis_lr

        ppo_train_round = 0

        for epi in range(self.args.episode):

            print('######################')

            start = time.time()

            reward_all = []
                
            with torch.no_grad():
                print('simulate!')
                self.policy_net.func.evnts = []
                self.policy_net.func.jump_type = 'simulate'

                tgrid = np.round(np.arange(0,self.config.traj_length,self.config.traj_length-1/self.args.shrink),decimals=self.args.decimals)

                c0_origin = [torch.randn(self.args.simulate_batch,self.args.dim_c).to(self.device) for i in range(3)]

                c0 = [self.policy_net.func.noise_linear(i) for i in c0_origin]

                h0 = [torch.zeros(self.args.simulate_batch,self.args.dim_h).to(self.device) for i in range(3)]

                # z0: the concatenation of three need levels
                z0 = torch.cat([torch.cat((c0[i],h0[i]),dim=-1) for i in range(3)],dim=-1)

                tsave = torch.FloatTensor(tgrid).to(self.device)

                _ = self.policy_net.act(z0,tsave)

                self.policy_net.func.evnts.sort()

                tevnt = np.array([i[0] for i in self.policy_net.func.evnts])

                tevolve = np.sort(np.unique(np.concatenate((tgrid,tevnt))))

                t2tid = {t: tid for tid, t in enumerate(tevolve)}

                etid = [t2tid[t] for t in tevnt] 

                tevolve = torch.tensor(tevolve,dtype=torch.float64).to(self.device)

                self.policy_net.func.jump_type = 'read'

                z = odeint(self.policy_net.func,z0,tevolve,method=self.args.ODE_method, rtol=self.args.rtol, atol=self.args.atol)

                week_hour_emb = self.t2weekhour_batch(tevolve,z.shape[1])

                z_week_hour = torch.cat((z ,week_hour_emb),dim=-1)

                Lambda = self.policy_net.func.L(z_week_hour)

                assert len(etid) == len(self.policy_net.func.evnts), 'evnts error!'

                self.policy_net.func.evnts.sort(key=lambda x:(x[1],x[0]))

                evnts_num = len(self.policy_net.func.evnts)/(self.args.simulate_batch+1e-9)

                for index, evnt in enumerate(self.policy_net.func.evnts):

                    tse = (t2tid[evnt[0]],) + evnt[1:]

                    if index==0 or evnt[0]<self.policy_net.func.evnts[index-1][0]:
                        continue # remove the first event

                    done = True if (index==len(self.policy_net.func.evnts)-1 or evnt[1] != self.policy_net.func.evnts[index+1][1]) else False

                    param = Lambda[tse[:2]]

                    param = torch.distributions.Categorical(param.float())
                    action_log_prob = param.log_prob(torch.tensor(evnt[2]).to(self.device)).cpu().detach().item()
                    action_dist_entropy = param.entropy().mean().cpu().detach().item()

                    state = [(i[0],i[2],self.config.Pid2N[i[2]],self.t2weekhour(i[0])[0],self.t2weekhour(i[0])[1],i[0]) for i in self.policy_net.func.evnts[:index] if i[1]==evnt[1]] # past evnts set

                    state = [(i[0] if i_index==0 else i[0]-state[i_index-1][0],i[1],i[2],i[3],i[4],i[5]) for i_index, i in enumerate(state)]

                    weekday, hour = self.t2weekhour(evnt[0])

                    action = (evnt[0]-self.policy_net.func.evnts[index-1][0],evnt[2],self.config.Pid2N[evnt[2]],weekday,hour,evnt[0]) # interval action, need, weekday, hour, t
                    
                    a1, a2, a3, a4, a5, _, _, _, _, _, a6 = self.state_action_tensor(state,action)

                    value = self.value_net(a1.to(self.device), a2.to(self.device), a3.to(self.device), a4.to(self.device), a5.to(self.device), a6.to(self.device)).cpu().detach().item()

                    reward = self.get_reward(state,action)

                    reward_all.append(reward)

                    c0_save = [i[evnt[1]].cpu().detach() for i in c0_origin]
                    h0_save = [i[evnt[1]].cpu().detach() for i in h0]

                    self.buffer.store(state, action, reward, done, value, action_log_prob, action_dist_entropy,(c0_save,h0_save))

            print('time of generating {} sequences: {}'.format(self.args.simulate_batch,(time.time()-start)/60.0))
            
            print('buffer length: {}'.format(len(self.buffer)))

            if len(self.buffer)>self.args.buffer_capacity:

                reward_all = []

                ppo_train_round += 1

                print('ppo train: {} round !'.format(ppo_train_round))

                self.buffer.compute_returns()

                self.ppo_train()
                print('time of ppo training: {}'.format((time.time()-start)/60.0))

                self.buffer.clear()

            print('discriminator buffer length: {}'.format(len(self.buffer.dis_memory)))

            if self.args.disc_train == 0:
                self.buffer.dis_memory.clear()

            else:
                if ppo_train_round % self.args.ppo_disc ==0:

                    self.discriminator_train(Type='generate')
                    self.buffer.dis_memory.clear()

            if ppo_train_round % self.args.evaluation_interval == 0:

                poi_distinct, event_cnt, interval, poi_cate, need, week, hour, poi_cnt, diversity,poi_interval,need_interval = self.evaluation_JSD()

                print('\n################# Evaluation! ################')

                a = [poi_distinct_max,event_cnt_max, interval_max, poi_cate_max, need_max, week_max, hour_max, poi_cnt_max,diversity_max]
                b = [poi_distinct, event_cnt, interval, poi_cate, need, week, hour, poi_cnt,diversity]

                poi_distinct_max,event_cnt_max, interval_max, poi_cate_max, need_max, week_max, hour_max, poi_cnt_max, diversity_max = min(poi_distinct_max,poi_distinct),min(event_cnt_max,event_cnt), min(interval_max,interval),min(poi_cate_max,poi_cate),min(need_max,need),min(week_max,week),min(hour_max,hour), min(poi_cnt_max,poi_cnt), min(diversity_max, diversity)

                self.policy_net.func.evnts = []

                torch.save(self.policy_net.state_dict(), self.model_path+'policy_net_{}.pkl'.format(epi))
