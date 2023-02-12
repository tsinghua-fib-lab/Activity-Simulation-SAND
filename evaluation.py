import numpy as np
import scipy.stats
import pandas as pd
import math
from bisect import bisect_right

class Evaluation(object):

    def __init__(self, config, args):
        self.config = config
        self.args= args

    def arr_to_distribution(self,arr, Min, Max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        if max(arr)<=Max:
            distribution, base = np.histogram(arr,bins = bins,range=(Min,Max))
        else:
            distribution, base = np.histogram(arr[arr<=Max],bins = bins,range=(Min,Max))
            m = np.array([len(arr[arr>Max])],dtype='int64')
            distribution = np.hstack((distribution,m))

        return distribution, base[:-1]

    def get_js_divergence(self, p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-9)
        p2 = p2 / (p2.sum()+1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def x2grid(self,x,Type='time'):
        if Type=='time':
            return bisect_right(self.config.time_grid,x)
        if Type=='poi':
            return bisect_right(self.config.poi_grid,x)

    def get_num(self,x):
        return [len(set([i[1] for i in u])) for u in x]

    def get_total_num(self,x):
        return [round(len(i)/(self.config.traj_length*self.args.shrink/24.0)) for i in x]

    def get_cate(self,x,Type='poi'):
        if Type=='poi':
            t = [i[1] for u in x for i in u]
            M = len(self.config.Pid2N)
        elif Type=='need':
            t = [i[2] for u in x for i in u]
            M = 3
        elif Type=='week':
            t = [i[3] for u in x for i in u]
            M = self.config.total_weekdays
        elif Type=='hour':
            t = [i[4] for u in x for i in u]
            M = self.config.total_hours
        t = pd.value_counts(t)
        tt = [[t.keys()[i],t.values[i]] for i in range(len(t))]
        for index in range(M):
            if index not in t.keys():
                tt.append([index,0])
        tt.sort()
        return np.array([i[1] for i in tt])

    def diversity_cal(self,x):
        t = pd.value_counts(x,normalize=True).values
        return -np.sum(t*np.log(t))


    def poi_interval(self,x): 
        interval = [i-x[index-1] for index, i in enumerate(x) if index>0]
        return np.mean(interval)

    def data_agg(self,seq):
        y = []
        for index,s in enumerate(seq):
            if index==0 or s[1]!=y[-1][1]:
                y.append(s)
        result = [[s[0] if index==0 else s[0]-y[index-1][0]]+s[1:] for index, s in enumerate(y)]
        return result

    def get_JSD(self,real,fake_origin,JSD_type='0'):

        if self.args.agg == 1:
            fake = [self.data_agg(seq) for seq in fake_origin]
        else:
            fake = fake_origin

        if JSD_type in ['0','1','2','7','8']:

            if JSD_type=='0': # number of distint activities
                r = self.get_num([u[1:] for u in real])
                f = self.get_num([u[1:] for u in fake])

            if JSD_type=='1': # number of activities
                r = self.get_total_num([u[1:] for u in real])
                f = self.get_total_num([u[1:] for u in fake])

            if JSD_type=='7': 
                r = [self.x2grid(len(u[1:]),'poi') for u in real]
                f = [self.x2grid(len(u[1:]),'poi') for u in fake]

            if JSD_type=='2': 
                r = [self.x2grid(i[0],'time') for u in real for i in u[1:]]
                f = [self.x2grid(i[0],'time') for u in fake for i in u[1:]]

            if JSD_type=='8': 
                r = [self.diversity_cal([i[1] for i in u[1:]]) for u in real]
                f = [self.diversity_cal([i[1] for i in u[1:]]) for u in fake]

            MIN = 0
            MAX = max(r+f)
            bins = math.ceil(MAX-MIN)

            if JSD_type=='8':
                bins = 5

            r_list, _ = self.arr_to_distribution(r, MIN, MAX, bins)
            f_list, _ = self.arr_to_distribution(f, MIN, MAX, bins)

            JSD = self.get_js_divergence(r_list, f_list)

        elif JSD_type in ['3','4','5','6']:

            if JSD_type=='3':
                r_list = self.get_cate([u[1:] for u in real],Type='poi')
                f_list = self.get_cate([u[1:] for u in fake],Type='poi')

            elif JSD_type == '4': 
                r_list = self.get_cate([u[1:] for u in real],Type='need')
                f_list = self.get_cate([u[1:] for u in fake],Type='need')

            elif JSD_type=='5':
                r_list = self.get_cate([u[1:] for u in real],Type='week')
                f_list = self.get_cate([u[1:] for u in fake],Type='week')

            elif JSD_type=='6':
                r_list = self.get_cate([u[1:] for u in real],Type='hour')
                f_list = self.get_cate([u[1:] for u in fake],Type='hour')

            JSD = self.get_js_divergence(r_list, f_list)
        
        elif JSD_type == '9':
            JSD = []
            for pid in range(len(self.config.Pid2N)):
                real_select = [[t[-1] for t in u if t[1]==pid] for u in real if len([i for i in u if i[1]==pid])>=2]
                fake_select = [[t[-1] for t in u if t[1]==pid] for u in fake if len([i for i in u if i[1]==pid])>=2]
            

                r = [self.x2grid(self.poi_interval(u),'time') for u in real_select]
                f = [self.x2grid(self.poi_interval(u),'time') for u in fake_select]

                if len(r)==0 or len(f)==0:
                    JSD.append(-1)
                    continue

                MIN = 0
                MAX = 8
                bins = math.ceil(MAX-MIN)

                r_list, _ = self.arr_to_distribution(r, MIN, MAX, bins)
                f_list, _ = self.arr_to_distribution(f, MIN, MAX, bins)

                JSD_t = self.get_js_divergence(r_list, f_list)

                JSD.append(JSD_t)
        
        elif JSD_type == '10':
            JSD = []
            for nid in range(3):
                real_select = [[t[-1] for t in u if t[2]==nid] for u in real if len([i for i in u if i[2]==nid])>=2]
                fake_select = [[t[-1] for t in u if t[2]==nid] for u in fake if len([i for i in u if i[2]==nid])>=2]

                r = [self.x2grid(self.poi_interval(u),'time') for u in real_select]
                f = [self.x2grid(self.poi_interval(u),'time') for u in fake_select]

                if len(r)==0 or len(f)==0:
                    JSD.append(-1)
                    continue

                MIN = 0
                MAX = 8
                bins = math.ceil(MAX-MIN)

                r_list, _ = self.arr_to_distribution(r, MIN, MAX, bins)
                f_list, _ = self.arr_to_distribution(f, MIN, MAX, bins)

                JSD_t = self.get_js_divergence(r_list, f_list)

                JSD.append(JSD_t)

        return JSD