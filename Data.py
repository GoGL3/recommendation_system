import os
import glob
import re
from itertools import chain
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt


DIR = 'data/'


class Data:

    def __init__(self, DIR):
        """
        :objective: load raw data
        :param: DIR = data location
        """
        self.DIR = DIR

        ## META
        self.meta = pd.read_json(self.DIR + 'metadata.json', lines=True)
        # POSIX timestamp 변환
        atc = self.meta.copy()
        atc['reg_datetime'] = atc['reg_ts'].apply(lambda x : datetime.fromtimestamp(x/1000.0))
        atc.loc[atc['reg_datetime'] == atc['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)
        atc['reg_dt'] = atc['reg_datetime'].dt.date
        atc['type'] = atc['magazine_id'].apply(lambda x : '개인' if x == 0.0 else '매거진')
        self.atc = atc


        ## USERS
        self.users = pd.read_json(self.DIR + '/users.json', lines = True)

        ## READ
        self.read_file_lst = glob.glob(self.DIR +'/read/*')

        exclude_file_lst = ['read.tar']
        read_df_lst = []
        for f in self.read_file_lst:
            file_name = os.path.basename(f)
            if file_name in exclude_file_lst:
                print(file_name)
            else:
                df_temp = pd.read_csv(f, header=None, names=['raw'])
                df_temp['dt'] = file_name[:8]
                df_temp['hr'] = file_name[8:10]
                df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0]
                df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip()
                read_df_lst.append(df_temp)
        read = pd.concat(read_df_lst)
        self.read = read

    #def get_users_follow(self):
    #    """
    #    :objective: get users / following list
    #    """
    #    self.users_follow = self.users['following_list'].map(len)
    #    return self.users_follow

    def preprocess_user_item(self):
        """
        :objective: process user and read data
        """
        user_follow = self.users['following_list'].map(len)
        self.user_pr = pd.DataFrame({'keyword_list': np.repeat(self.users['keyword_list'], user_follow),
                                 'id': np.repeat(self.users['id'], user_follow),
                                 'following_list': self.chainer(self.users['following_list'],'user')})

        read_cnt_by_user = self.read['article_id'].str.split(' ').map(len)
        self.read_pr = pd.DataFrame({'dt': np.repeat(self.read['dt'], read_cnt_by_user),
                                 'hr': np.repeat(self.read['hr'], read_cnt_by_user),
                                 'user_id': np.repeat(self.read['user_id'], read_cnt_by_user),
                                 'article_id': self.chainer(self.read['article_id'],'read')})

    def chainer(self, s, type):
        """
        :objective: expand elements in list to separate rows
        :param: type = 'user'/'read'
        """
        if type == 'user':
            return list(chain.from_iterable(s))
        else:
            return list(chain.from_iterable(s.str.split(' ')))


    def get_df_user_view_cnt(self):
        """
        :objective: expand elements in list to separate rows
        :param: type = 'user'/'read'
        """
        read_view = self.read_pr.copy()
        self.df_user_view_cnt = read_view.groupby(by=['user_id','article_id'], as_index=False).count()
        #df_user_view_day_cnt = read_view.groupby(by=['user_id','article_id','dt'], as_index=False).count()
        #df_user_day_cnt = read_view.groupby(by=['user_id','dt'], as_index=False).count()

    def save_mapping(self, save=False):
        """
        :objective: expand elements in list to separate rows
        :param: type = 'user'/'read'
        """
        user_map = self.create_mapping(self.df_user_view_cnt['user_id'], "user_map.csv")
        article_map = self.create_mapping(self.df_user_view_cnt['article_id'], "article_map.csv")
        df = self.df_user_view_cnt.copy()
        df["userId"] = df["user_id"].map(user_map.get)
        df["itemId"] = df["article_id"].map(article_map.get)
        self.mapped_df = df[["userId", "itemId", "dt"]]

        if save:
            self.mapped_df.to_csv(path_or_buf = "collab_mapped.csv", index = False, header = False)


    def create_mapping(self,values,filename):
        """
        :objective: expand elements in list to separate rows
        :param: type = 'user'/'read'
        """
        with open(filename,'w') as ofp:
            value_to_id = {value:idx for idx, value in enumerate(values.unique())}
            for value, idx in value_to_id.items():
                ofp.write('{} {}\n'.format(value, idx))
        return value_to_id














## Transform raw data into train/test

def train_test_split()


def get_train_instances()

###############################
##       RUN FROM HERE       ##
###############################
mapped_df = pd.read_csv(DIR + 'collab_mapped.csv')

mapped_toyy = mapped_df.copy()
mapped_toyy['itemId'] = mapped_toyy['itemId'].astype(int)
mapped_toyy['userId'] = mapped_toyy['userId'].astype(int)

# mapped_toy = mapped_toyy.sort_values(by=['dt'],axis=0,ascending=False)
mapped_toy = mapped_toyy[(mapped_toyy.dt <=100) & (mapped_toyy.dt >=10)]
mapped_toy.shape

df_train, df_test = train_test_split(mapped_toy)
df_train.shape
df_test.shape

users = list(np.sort(df.userId.unique()))
items = list(np.sort(df.itemId.unique()))

rows = df_train.userId.astype(int)
cols = df_train.itemId.astype(int)
values = list(df_train.dt)

uids = np.array(rows.tolist())
iids = np.array(cols.tolist())


from sklearn.utils import shuffle
from Metric import Metric
from NeuMF import NeuMF


user_input, item_input, labels = get_train_instances(uids, iids, num_neg, len(items))
user_data_shuff, item_data_shuff, label_data_shuff = shuffle(user_input, item_input, labels)
user_data_shuff = np.array(user_data_shuff).reshape(-1,1)
item_data_shuff = np.array(item_data_shuff).reshape(-1,1)
label_data_shuff = np.array(label_data_shuff).reshape(-1,1)

nmf = NeuMF(len(users), len(items))  # Neural Collaborative Filtering
model = nmf.get_model()
model.fit([user_data_shuff, item_data_shuff], label_data_shuff, epochs=5,
                       batch_size=256, verbose=1)
