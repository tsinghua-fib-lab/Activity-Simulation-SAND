import argparse
import os
import copy
from Trainer import Trainer
from config import config
import numpy as np
import torch
import random
import json

from config import config

def parse_args():
    parser = argparse.ArgumentParser(description="Training Parameter")

    # initialize the experiment
    parser.add_argument('--Model', type=str, default='SAND', help='Model')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--is_pretrain',type=int,default=0)
    parser.add_argument('--cuda_id', type=int, default=0, help='gpu id')
    parser.add_argument('--SEED', type=int, default=520, help='random seed')
    parser.add_argument('--shrink', type=float, default=1000, help='None')
    parser.add_argument('--decimals', type=int, default=3, help='')
    parser.add_argument('--generate_final_path', type=str, default='../gene_data/', help='')
    parser.add_argument('--trained_model', type=str, default='../trained_model/', help='')
    parser.add_argument('--agg', type=int, default=1, help='aggregation')

    # ODE parameter
    parser.add_argument('--ODE_method', type=str, default='jump_adams', help='')
    parser.add_argument('--rtol', type=float, default=1.0e-5, help='None')
    parser.add_argument('--atol', type=float, default=1.0e-7, help='None')
    
    # embedding dimension
    parser.add_argument('--dim_c', type=int, default=4, help='None')
    parser.add_argument('--dim_h', type=int, default=4, help='None')
    parser.add_argument('--hour_dim', type=int, default=8, help='None')
    parser.add_argument('--week_dim', type=int, default=4, help='None')
    parser.add_argument('--dim_hidden', type=int, default=32, help='None')
    parser.add_argument('--dis_dim_hidden', type=int, default=8, help='None')
    parser.add_argument('--num_hidden', type=int, default=2, help='None')
    parser.add_argument('--f_embedding_dim', type=int, default=8, help='None')
    parser.add_argument('--d_embedding_dim', type=int, default=8, help='None')

    # learning rate and weight decay
    parser.add_argument('--policy_lr', type=float, default=3e-4, help='None')
    parser.add_argument('--pretrain_lr1', type=float, default=3e-4, help='None')
    parser.add_argument('--pretrain_lr2', type=float, default=3e-4, help='None')
    parser.add_argument('--value_lr', type=float, default=3e-4, help='None')
    parser.add_argument('--dis_lr', type=float, default=1e-4, help='None')
    parser.add_argument('--dis_lr_pretrain', type=float, default=5e-3, help='None')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='None')

    # training epoches, batch size, buffer size
    parser.add_argument('--simulate_batch', type=int, default=64, help='None')
    parser.add_argument('--buffer_capacity', type=int, default=128, help='None')
    parser.add_argument('--episode', type=int, default=2000, help='None')
    parser.add_argument('--dis_iter', type=int, default=4, help='None')
    parser.add_argument('--ppo_disc', type=int, default=5, help='None')
    parser.add_argument('--value_iter', type=int, default=10, help='None')
    parser.add_argument('--ppo_epoch', type=int, default=8, help='None')
    parser.add_argument('--ODE_pretrain_batch', type=int, default=100, help='None')
    parser.add_argument('--dis_batch', type=int, default=50000, help='None')
    parser.add_argument('--ODE_pretrain_epoch', type=int, default=20, help='None')
    parser.add_argument('--ODE_pretrain_expert', type=int, default=2000, help='None')
    parser.add_argument('--test_batch', type=int, default=500, help='None')
    parser.add_argument('--generate_cnt', type=int, default=2000, help='None')
    parser.add_argument('--evaluation_interval', type=int, default=5, help='')
    parser.add_argument('--generate_num_down', type=int, default=2000, help='None')

    # hyper parameter
    parser.add_argument('--gamma', type=float, default=0.99, help='None')
    parser.add_argument('--lmbda', type=float, default=0.95, help='None')
    parser.add_argument('--ortho', type=int, default=1, help='Bool type')
    parser.add_argument('--entropy_coef', type=float, default=0.0, help='None')
    parser.add_argument('--entropy_decay', type=float, default=0.5, help='None')
    parser.add_argument('--coef_decay_epoch', type=int, default=0, help='None')

    #policy network
    parser.add_argument('--adv_norm', type=int, default=1, help='None')
    parser.add_argument('--disc_train', type=int, default=1, help='None')
    
    # clip
    parser.add_argument('--clip_grad', type=float, default=2, help='None')
    parser.add_argument('--clip_param', type=float, default=0.2, help='None')
    parser.add_argument('--disc_clip', type=int, default=1, help='None')

    # dataset
    parser.add_argument('--dataset', type=str, default='Mobile', help='dataset selection',choices=['Mobile','Foursquare','Synthetic'])
    parser.add_argument('--fake_file', type=str, default='mobile', help='')
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)

def seed_torch(seed=args.SEED):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def fake_generate(config_len=24*14,len_avg = 20, dif = 5, n_fake=1000,n_poi=17):
    lens = []
    result = []
    for _ in range(n_fake):
        temp = []
        l = random.randint(round(len_avg-dif), round(len_avg+dif))
        lens.append(l)
        for _ in range(l):
            t = random.random()*config_len
            poi = random.randint(0, n_poi-1)
            temp.append([t,poi])
        temp.sort()
        result.append(temp)
    return result

seed_torch()

if __name__ == '__main__':

    if args.dataset == 'Mobile':
        real_file_path = config.real_data_mobile
        config.P2id = config.mobile_P2id
        config.Pid2N = config.mobile_Pid2N
        config.traj_length = 336
        args.traj_length = 336
    
    elif args.dataset == 'Foursquare':
        real_file_path = config.real_data_foursquare
        config.P2id = config.foursquare_P2id
        config.Pid2N = config.foursquare_Pid2N
        config.traj_length = 672
        args.traj_length = 672

    elif args.dataset == 'Synthetic':
        real_file_path = config.synthetic_data
        config.P2id = config.synthetic_P2id
        config.Pid2N = config.synthetic_Pid2N
        config.traj_length = 72
        args.traj_length = 72

    with open(real_file_path,'r') as f:
        all_file = json.load(f)
        random.shuffle(all_file)

    all_file = [[[i[0],config.P2id[i[1]]] for i in u if i[0]<=config.traj_length] for u in all_file]
    all_file = [[[np.round(i[0]/args.shrink,args.decimals),i[1]] for i in u] for u in all_file]

    config.time_grid = [i/args.shrink for i in config.time_grid]
    config.traj_length /= args.shrink

    config.real_len_avg = np.mean([len(i) for i in all_file])
    config.std = np.std([len(i) for i in all_file])

    test_file_origin = all_file[int(len(all_file)*0.8):]
    real_file = all_file[:int(len(all_file)*0.8)]

    test_file = []
    for seq in test_file_origin:
        test_file.append([[i[0] if index==0 else i[0]-seq[index-1][0],i[1],config.Pid2N[i[1]],int(i[0]*args.shrink/24)%7,int(i[0]*args.shrink%24),i[0]] for index, i in enumerate(seq)])

    fake = fake_generate(config.traj_length,config.real_len_avg,config.std,1000,len(config.P2id))

    fake_file = []
    for seq in fake:
        fake_file.append([[i[0] if index==0 else i[0]-seq[index-1][0],i[1],config.Pid2N[i[1]],int(i[0]*args.shrink/24)%7,int(i[0]*args.shrink%24)] for index, i in enumerate(seq)])

    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    app = Trainer(config, args, real_file, test_file, fake_file, device)
    if args.mode == 'train':
        app.overall_train()
    elif args.mode=='generate':
        app.test_generate()
