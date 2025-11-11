# import os
# import numpy as np
# import pandas as pd
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', cycle=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         print(3*5)
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cycle = cycle
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         # train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
#         # n = len(train_data_std)
#         # # 位置编号（考虑偏移）
#         # pos = (np.arange(n)) % self.cycle
#         # # 按“周期内位置”分组取均值；自动对 NaN 做跳过（mean 的默认行为）
#         # out = train_data_std.groupby(pos).mean()
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#         # add cycle
#         self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
#         self.cycle_data = out
#         print('self.cycle_data', self.cycle_data)
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         cycle_index = torch.tensor(self.cycle_index[s_end])
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=True, timeenc=0, freq='t', cycle=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cycle = cycle
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#         # add cycle
#         self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         cycle_index = torch.tensor(self.cycle_index[s_end])
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', cycle=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cycle = cycle
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             # print(self.scaler.mean_)
#             # exit()
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#         # add cycle
#         self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         cycle_index = torch.tensor(self.cycle_index[s_end])
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# ## TODO add cycle
# class Dataset_Pred(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['pred']
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols = cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         if self.cols:
#             cols = self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns)
#             cols.remove(self.target)
#             cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         border1 = len(df_raw) - self.seq_len
#         border2 = len(df_raw)
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             self.scaler.fit(df_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         tmp_stamp = df_raw[['date']][border1:border2]
#         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
#
#         df_stamp = pd.DataFrame(columns=['date'])
#         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = self.data_x[r_begin:r_begin + self.label_len]
#         else:
#             seq_y = self.data_y[r_begin:r_begin + self.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_Solar(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, cycle=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cycle = cycle
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = []
#         with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
#             for line in f.readlines():
#                 line = line.strip('\n').split(',')
#                 data_line = np.stack([float(i) for i in line])
#                 df_raw.append(data_line)
#         df_raw = np.stack(df_raw, 0)
#         df_raw = pd.DataFrame(df_raw)
#
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_valid = int(len(df_raw) * 0.1)
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_valid, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         df_data = df_raw.values
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data)
#             data = self.scaler.transform(df_data)
#         else:
#             data = df_data
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#
#         # add cycle
#         self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1))
#         seq_y_mark = torch.zeros((seq_x.shape[0], 1))
#
#         cycle_index = torch.tensor(self.cycle_index[s_end])
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_PEMS(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', cycle=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cycle = cycle
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         data_file = os.path.join(self.root_path, self.data_path)
#         data = np.load(data_file, allow_pickle=True)
#         data = data['data'][:, :, 0]
#
#         num_train = int(len(data) * 0.6)
#         num_test = int(len(data) * 0.2)
#         num_valid = int(len(data) * 0.2)
#         border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_valid, len(data)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.scale:
#             train_data = data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data)
#             data = self.scaler.transform(data)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#
#         # add cycle
#         self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1))
#         seq_y_mark = torch.zeros((seq_x.shape[0], 1))
#
#         cycle_index = torch.tensor(self.cycle_index[s_end])
#         return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 对训练集每个通道做移动平均分解（trend = rolling mean），得到季节项 seasonal = original - trend
        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        # 使用窗口为 self.cycle 的居中移动平均估计趋势
        trend = train_data_std.rolling(window=int(self.cycle), center=True, min_periods=1).mean()
        print('cycle', self.cycle, 'train_data_std', train_data_std.shape, 'trend', trend.shape)
        seasonal = train_data_std #- trend

        # 按“周期内位置”分组取季节项的均值，得到 cycle_data
        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = seasonal.groupby(pos).mean()

        # train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        # n = len(train_data_std)
        # # 位置编号（考虑偏移）
        # pos = (np.arange(n)) % self.cycle
        # # 按“周期内位置”分组取均值；自动对 NaN 做跳过（mean 的默认行为）
        # out = train_data_std.groupby(pos).mean()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['day'] = df_stamp['date'].dt.weekday
        self.hour_index = df_stamp['hour'].values
        self.day_index = df_stamp['day'].values

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values
        # print('self.cycle_data', self.cycle_data.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 对训练集每个通道做移动平均分解（trend = rolling mean），得到季节项 seasonal = original - trend
        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        # 使用窗口为 self.cycle 的居中移动平均估计趋势
        trend = train_data_std.rolling(window=int(self.cycle), center=True, min_periods=1).mean()

        print('cycle', self.cycle, 'train_data_std', train_data_std.shape, 'trend', trend.shape)
        seasonal = train_data_std #- trend

        # 按“周期内位置”分组取季节项的均值，得到 cycle_data
        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = seasonal.groupby(pos).mean()

        # train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        # n = len(train_data_std)
        # # 位置编号（考虑偏移）
        # pos = (np.arange(n)+self.seq_len) % self.cycle
        # # 按“周期内位置”分组取均值；自动对 NaN 做跳过（mean 的默认行为）
        # out = train_data_std.groupby(pos).mean()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['day'] = df_stamp['date'].dt.weekday
        self.hour_index = df_stamp['hour'].values
        self.day_index = df_stamp['day'].values

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 对训练集每个通道做移动平均分解（trend = rolling mean），得到季节项 seasonal = original - trend
        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        # 使用窗口为 self.cycle 的居中移动平均估计趋势
        trend = train_data_std.rolling(window=int(self.cycle), center=True, min_periods=1).mean()

        print('cycle', self.cycle, 'train_data_std', train_data_std.shape, 'trend', trend.shape)
        seasonal = train_data_std #- trend

        # 按“周期内位置”分组取季节项的均值，得到 cycle_data
        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = seasonal.groupby(pos).mean()
        # out2 = train_data_std.groupby(pos).mean()
        # print(train_data_std.max(), train_data_std.min(), train_data_std.mean())
        # print('====')
        # diff = out - out2
        # print(diff.max(), diff.min(), diff.mean())
        # exit()

        # train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        # n = len(train_data_std)
        # # 位置编号（考虑偏移）
        # pos = (np.arange(n)) % self.cycle
        # 按“周期内位置”分组取均值；自动对 NaN 做跳过（mean 的默认行为）
        # out2 = train_data_std.groupby(pos).mean()
        # diff = out - out2
        # print(train_data_std.max(), train_data_std.min(), train_data_std.mean())
        # print('====')
        # print(diff.max(), diff.min(), diff.mean())
        # exit()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['day'] = df_stamp['date'].dt.weekday
        df_stamp['month'] = df_stamp['date'].dt.month
        df_stamp['day_in_month'] = df_stamp['date'].dt.day
        self.hour_index = df_stamp['hour'].values
        self.day_index = df_stamp['day'].values
        self.month_index = df_stamp['month'].values - 1  # 0-11
        self.day_in_month_index = df_stamp['day_in_month'].values - 1  # 0-30

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

        # # 保存 cycle_data 到 .npy（位于 root_path）
        # try:
        #     basename = os.path.splitext(self.data_path)[0] if hasattr(self, 'data_path') else 'data'
        #     print(123)
        #     save_name = f'cycle_data_{basename}_cycle{int(self.cycle)}.npy'
        #     save_path = os.path.join(self.root_path, save_name)
        #     np.save(save_path, self.cycle_data)
        # except Exception as e:
        #     print(f'Warning: failed to save cycle_data to .npy: {e}')

        # # out 是上面计算得到的 pandas DataFrame（index 为周期内位置，columns 为通道）
        # with PdfPages(pdf_path) as pdf:
        #     for idx, col in enumerate(out.columns):
        #         fig, ax = plt.subplots(figsize=(8, 4))
        #         ax.plot(out.index, out[col].values, label=f'decomp_{col}', color='C0', linestyle='--')
        #         ax.plot(out.index, out2[col].values, label=f'no-decomp_{col}', color='C1', linestyle='-')
        #         ax.set_title(f'Cycle mean - {col}')
        #         ax.set_xlabel('position_in_cycle')
        #         ax.set_ylabel('seasonal_value')
        #         ax.grid(True)
        #         ax.legend()
        #         plt.tight_layout()
        #         pdf.savefig(fig)
        #         plt.close(fig)
        #
        # pdf_name = f'cycle_data_{basename}_cycle{int(self.cycle)}_3.pdf'
        # pdf_path = os.path.join(self.root_path, pdf_name)
        # # out_trained = np.load('dataset/trained_ECL.npy')
        #
        # # out 是上面计算得到的 pandas DataFrame（index 为周期内位置，columns 为通道）
        # with PdfPages(pdf_path) as pdf:
        #     for idx, col in enumerate(out.columns):
        #         fig, ax = plt.subplots(figsize=(8, 4))
        #         ax.plot(train_data_std[col].values[:1000], label=f'train_data_std_{col}', color='C0', linestyle='--')
        #         ax.plot(trend[col].values[:1000], label=f'trend_{col}', color='C1', linestyle='-')
        #         ax.set_title(f'Decomp Result - {col}')
        #         ax.set_xlabel('t')
        #         ax.set_ylabel('value')
        #         ax.grid(True)
        #         ax.legend()
        #         plt.tight_layout()
        #         pdf.savefig(fig)
        #         plt.close(fig)
        # exit()

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # from matplotlib.backends.backend_pdf import PdfPages
        #
        # pdf_name = f'cycle_data_{basename}_cycle{int(self.cycle)}.pdf'
        # pdf_path = os.path.join(self.root_path, pdf_name)
        # out_trained = np.load('dataset/trained_ECL.npy')
        #
        # # out 是上面计算得到的 pandas DataFrame（index 为周期内位置，columns 为通道）
        # with PdfPages(pdf_path) as pdf:
        #     for idx, col in enumerate(out.columns):
        #         fig, ax = plt.subplots(figsize=(8, 4))
        #         ax.plot(out.index, out[col].values, label=f'fixed_{col}', color='C0', linestyle='--')
        #         ax.plot(out.index, out_trained[:, idx], label=f'train_{col}', color='C1', linestyle='-')
        #         ax.set_title(f'Cycle mean - {col}')
        #         ax.set_xlabel('position_in_cycle')
        #         ax.set_ylabel('seasonal_value')
        #         ax.grid(True)
        #         plt.tight_layout()
        #         pdf.savefig(fig)
        #         plt.close(fig)
        #
        # # 将 out 按训练集长度展开（pos 映射），然后和 train_data_std 的相同通道一起画图并保存到 PDF
        # try:
        #     # out.values 形状 (cycle, C)，pos 长度为 n -> out_long (n, C)
        #     out_long = out.values[pos]
        #     out_trained_long = out_trained[pos]
        #
        #     import matplotlib
        #     matplotlib.use('Agg')
        #     import matplotlib.pyplot as plt
        #     from matplotlib.backends.backend_pdf import PdfPages
        #
        #     pdf_name = f'cycle_data_{basename}_cycle{int(self.cycle)}_1.pdf'
        #     pdf_path = os.path.join(self.root_path, pdf_name)
        #
        #     with PdfPages(pdf_path) as pdf:
        #         for idx, col in enumerate(out.columns):
        #             fig, ax = plt.subplots(figsize=(10, 4))
        #             # 训练集原始通道序列
        #             ax.plot(np.arange(n)[:1000], train_data_std[col].values[:1000], label=f'train_{col}', color='C0', alpha=0.8)
        #             # 展开的周期均值
        #             ax.plot(np.arange(n)[:1000], out_long[:1000, idx], label=f'cycle_mean_repeated_{col}', color='C1', linestyle='--', alpha=0.9)
        #             ax.plot(np.arange(n)[:1000], out_trained_long[:1000, idx], label=f'cycle_mean_trained_{col}', color='C2', linestyle='--', alpha=0.9)
        #
        #             ax.set_title(f'Channel {col} - train vs repeated cycle mean')
        #             ax.set_xlabel('train_index')
        #             ax.set_ylabel('value')
        #             ax.legend()
        #             ax.grid(True)
        #             plt.tight_layout()
        #             pdf.savefig(fig)
        #             plt.close(fig)
        # except Exception as e:
        #     print(f'Warning: failed to save cycle_data to PDF: {e}')

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])
        month_index = torch.tensor(self.month_index[s_end])
        day_in_month_index = torch.tensor(self.day_in_month_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index, month_index, day_in_month_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


## TODO add cycle
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        # 对训练集每个通道做移动平均分解（trend = rolling mean），得到季节项 seasonal = original - trend
        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]])
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        # 使用窗口为 self.cycle 的居中移动平均估计趋势
        trend = train_data_std.rolling(window=int(self.cycle), center=True, min_periods=1).mean()

        print('cycle', self.cycle, 'train_data_std', train_data_std.shape, 'trend', trend.shape)
        seasonal = train_data_std #- trend

        # 按“周期内位置”分组取季节项的均值，得到 cycle_data
        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = seasonal.groupby(pos).mean()

        # train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]])
        # n = len(train_data_std)
        # # 位置编号（考虑偏移）
        # pos = (np.arange(n)+self.seq_len) % self.cycle
        # # 按“周期内位置”分组取均值；自动对 NaN 做跳过（mean 的默认行为）
        # out = train_data_std.groupby(pos).mean()

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_valid = int(len(data) * 0.2)
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        # 使用窗口为 self.cycle 的居中移动平均估计趋势
        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]])
        n = len(train_data_std)
        trend = train_data_std.rolling(window=int(self.cycle), center=True, min_periods=1).mean()

        print('cycle', self.cycle, 'train_data_std', train_data_std.shape, 'trend', trend.shape)
        seasonal = train_data_std #- trend

        # 按“周期内位置”分组取季节项的均值，得到 cycle_data
        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = seasonal.groupby(pos).mean()


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
