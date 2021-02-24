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
                df_temp['dt'] = datetime.strptime(file_name[2:10]+'0000', '%y%m%d%H%M%S')
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
        # User
        user_follow = self.users['following_list'].map(len)
        self.user_pr = pd.DataFrame({'keyword_list': np.repeat(self.users['keyword_list'], user_follow),
                                 'id': np.repeat(self.users['id'], user_follow),
                                 'following_list': self.chainer(self.users['following_list'],'user')})
        # Read
        # Contextual Features
        read_ctx = self.read.copy()
        read_ctx['year'] = read_ctx['dt'].apply(lambda x: parse(str(x)).year)
        read_ctx['month'] = read_ctx['dt'].apply(lambda x: parse(str(x)).month)
        read_ctx['weekday'] = read_ctx['dt'].apply(lambda x: parse(str(x)).weekday())
        read_ctx['daytime'] = read_ctx['dt'].apply(lambda x: self.get_time_of_day(parse(str(x)).hour)) # time of day
        # read_ctx['timepast'] = read_ctx['article_id'].apply(lambda x: self.get_timepast(x))

        read_cnt_by_user = self.read['article_id'].str.split(' ').map(len)
        self.read_pr = pd.DataFrame({'year': np.repeat(read_ctx['year'], read_cnt_by_user),
                                     'month': np.repeat(read_ctx['month'], read_cnt_by_user),
                                     'weekday': np.repeat(read_ctx['weekday'], read_cnt_by_user),
                                     'daytime': np.repeat(read_ctx['daytime'], read_cnt_by_user),
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

    def get_time_of_day(self, hour):
        """
        :objective: --'preprocess_user_item' FE
        """
        if 6<=hour<=11:
            time=1
        elif 11<hour<=16:
            time=2
        elif 16<hour<=21:
            time=3
        elif 21<hour<=2:
            time=4
        else:
            time=5
        return time

    def get_df_user_view_cnt(self):
        """
        :objective: create df where each row stands for one interaction
        """
        read_view = self.read_pr.copy()
        read_view['val'] = 1

        df_user_view_cnt = read_view.groupby(by=['user_id','article_id',
        "year","month","weekday","daytime"], as_index=False).count()
        self.df_user_view_cnt_drop = df_user_view_cnt.drop(df_user_view_cnt[df_user_view_cnt.val>20].index)
        self.df_user_view_cnt_drop=self.df_user_view_cnt_drop.drop(['val'], axis=1)
        self.df_user_view_cnt_drop['val']=1

        #df_user_view_day_cnt = read_view.groupby(by=['user_id','article_id','dt'], as_index=False).count()
        #df_user_day_cnt = read_view.groupby(by=['user_id','dt'], as_index=False).count()

    def save_mapping(self, save=False):
        """
        :objective: map complicated IDs into int
        :param: save
        """
        file_path1 = "user_map.csv"
        file_path2 = "article_map.csv"
        if (os.path.exists(file_path1)) & (os.path.exists(file_path2)):
            user_map = pd.read_csv(file_path1)
            article_map = pd.read_csv(file_path2)
        else:
            user_map = self.create_mapping(self.df_user_view_cnt['user_id'], "user_map.csv")
            article_map = self.create_mapping(self.df_user_view_cnt['article_id'], "article_map.csv")
        df = self.df_user_view_cnt.copy()
        df["userId"] = df["user_id"].map(user_map.get)
        df["itemId"] = df["article_id"].map(article_map.get)
        self.mapped_df = df[["userId", "itemId","year","month","weekday","daytime","val"]]

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

    def run_preprocess(self):
        """
        :objective: run all preprocessing steps
        """
        self.preprocess_user_item()
        self.chainer()
        self.get_time_of_day()

        self.get_df_user_view_cnt()
        self.save_mapping(save=True)
        self.create_mapping()


    def prepare_df(self):
        """
        :objective: prepare preprocessed data before creating test / train
                    -- because preprocessing step is very heavy, load already processed data (suggested)
        """
        file_path = 'collab_mapped.csv'
        if (os.path.exists(file_path)):
            df = pd.read_csv('read_pr.csv',names=['user_id','item_id','year','month','weekday','daytime','val'])
            print('Load preprocessed data')
        else:
            print('Preprocessing...might take hours')
            self.run_preprocess()
            df = self.mapped_df

        df_order = df.copy() ## order data by date / necessary for creating test data
        df_order = df_order.sort_values(by=['year','month','weekday','daytime'])

        # Create training and test sets.
        df_train, df_test = self.train_test_split(df_order)

        ##############
        # Create lists of all unique users and artists
        users = list(np.sort(df_order.user_id.unique()))
        items = list(np.sort(df_order.item_id.unique()))

        # Get the rows, columns and values for our matrix.
        rows = df_train['user_id'].astype(int)
        cols = df_train['item_id'].astype(int)

        values = list(self.df_train['val'])

        # Get all user ids and item ids.
        uids = np.array(rows.tolist())
        iids = np.array(cols.tolist())

        # Sample 100 negative interactions for each user in our test data
        df_neg = get_negatives(uids, iids, items, df_test)

        return uids, iids, df_train, df_test, df_neg, users, items


    def get_negatives(uids, iids, items, df_test):
        """
        :objective:: returns a pandas dataframe of 100 negative interactions
        based for each user in df_test.
        :args:
            uids (np.array): Numpy array of all user ids.
            iids (np.array): Numpy array of all item ids.
            items (list): List of all unique items.
            df_test (dataframe): Our test set.
        :returns:
            df_neg (dataframe): dataframe with 100 negative items
                for each (u, i) pair in df_test.
        """

        negativeList = []
        test_u = df_test['user_id'].values.tolist()
        test_i = df_test['item_id'].values.tolist()

        test_ratings = list(zip(test_u, test_i))
        zipped = set(zip(uids, iids))

        for (u, i) in test_ratings:
            negatives = []
            negatives.append((u, i))
            for t in range(100):
                j = np.random.randint(len(items)) # Get random item id.
                while (u, j) in zipped: # Check if there is an interaction
                    j = np.random.randint(len(items)) # If yes, generate a new item id
                negatives.append(j) # Once a negative interaction is found we add it.
            negativeList.append(negatives)

        df_neg = pd.DataFrame(negativeList)

        return df_neg


def mask_first(x):
    """
    Return a list of 0 for the first item and 1 for all others
    """
    result = np.ones_like(x)
    result[0] = 0

    return result

def train_test_split(df):
    """
    Splits our original data into one test and one
    training set.
    The test set is made up of one item for each user. This is
    our holdout item used to compute Top@K later.
    The training set is the same as our original data but
    without any of the holdout items.
    Args:
        df (dataframe): Our original data
    Returns:
        df_train (dataframe): All of our data except holdout items
        df_test (dataframe): Only our holdout items.
    """

    # Create two copies of our dataframe that we can modify
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # Group by user_id and select only the first item for
    # each user (our holdout).
    df_test = df_test.groupby(['user_id']).last()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'item_id', 'plays']]
    del df_test.index.name

    # Remove the same items as we for our test set in our training set.
    mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test














user_input, item_input, labels = get_train_instances(uids, iids, num_neg, len(items))
user_data_shuff, item_data_shuff, label_data_shuff = shuffle(user_input, item_input, labels)
user_data_shuff = np.array(user_data_shuff).reshape(-1,1)
item_data_shuff = np.array(item_data_shuff).reshape(-1,1)
label_data_shuff = np.array(label_data_shuff).reshape(-1,1)

nmf = NeuMF(len(users), len(items))  # Neural Collaborative Filtering
model = nmf.get_model()
model.fit([user_data_shuff, item_data_shuff], label_data_shuff, epochs=5,
                       batch_size=256, verbose=1)
