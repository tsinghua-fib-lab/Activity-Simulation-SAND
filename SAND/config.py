# -*- coding: utf-8 -*-

class CONFIG:
    def __init__(self):
        super(CONFIG,self).__init__()
        self.real_data_mobile = 'dataset/Beijing_mobile.json'
        self.real_data_foursquare = 'dataset/USA_Foursqaure.json'
        self.synthetic_data = 'dataset/synthetic_data.json'

        self.traj_length = 0

        self.total_hours = 24
        self.total_weekdays = 7

        self.time_res = 1 # 时间分辨率
        self.time_grid = [1,2,4,8,16,32,64,128]
        self.poi_grid = [10,15,20,25,30,35,40,45,50]

        self.mobile_P2id = {'住宅': 0, '政府团体': 1, '医疗服务': 2, '文化场馆': 3, '生活服务营业厅': 4, '教育培训': 5, '其它生活服务': 6, '旅游度假': 7, '购物': 8, '音乐厅': 9, '中小学幼儿园': 10, '小商品市场超市': 11, '休闲': 12, '大学': 13, '运动': 14, '公司企业': 15, '餐饮': 16}
        self.mobile_Pid2N = {0: 0, 1: 1, 2: 0, 3: 2, 4: 0, 5: 1, 6: 0, 7: 2, 8: 2, 9: 2, 10: 1, 11: 2, 12: 2, 13: 1, 14: 2, 15: 1, 16: 0}

        self.foursquare_P2id = {'Shop & Service': 0, 'Nightlife Spot': 1, 'Food': 2, 'School': 3, 'College & University': 4, 'Athletics & Sports': 5, 'Food & Drink Shop': 6, 'Office': 7, 'Arts & Entertainment': 8, 'Medical Center': 9, 'Movie Theater': 10, 'Clothing Store': 11, 'Outdoors & Recreation': 12, 'Professional & Other Places': 13}
        self.foursquare_Pid2N = {0: 0, 1: 2, 2: 0, 3: 1, 4: 1, 5: 2,  6: 2, 7: 1, 8: 2, 9: 0, 10: 2, 11: 2, 12: 2, 13: 1}

        self.synthetic_P2id = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7}
        self.synthetic_Pid2N ={0:0,1:0,2:1,3:1,4:2,5:2,6:2,7:2}

config = CONFIG()

