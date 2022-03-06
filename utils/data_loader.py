import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


class KnowData(Dataset):
    def __init__(self, year, mode, args, nth_test=None):
        self.DATA_ROOT = args['DATA_ROOT']  # root : /KNOW_data/
        self.str_idx = args['STRING_INDEX']
        self.fold = args['FOLD']
        self.mode = mode
        self.nth_test = nth_test

        # read csv
        if mode == 'test':
            df = pd.read_csv(self.DATA_ROOT / 'test/KNOW_{}_test.csv'.format(year))
        else:
            df = pd.read_csv(self.DATA_ROOT / 'train/KNOW_{}.csv'.format(year))

        def clean_string(x):
            try:
                int(x)
                return x
            except:
                return '0'

        for col in df.columns.values:
            df[col] = df[col].apply(clean_string)
        df.replace('없음', 0, inplace=True)
        df.replace(' ', 0, inplace=True)
        df.replace('', 0, inplace=True)
        df.fillna(0, inplace=True)

        df = df.drop(columns=self.str_idx)
        df = df.replace(' ', '0')
        self.feature_name = list(df.columns.values)
        self.np_data = np.array(df)  # data frame to numpy

        self._preprocess()

    def _data_interval(self, tot_num):
        return list(range(int(tot_num * (self.nth_test - 1) / self.fold), int(tot_num * self.nth_test / self.fold)))

    def _preprocess(self):
        if self.mode != 'test':
            # encoding label
            labels = self.np_data[:, -1:].reshape(-1)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)

            # preprocess data
            num_data = self.np_data[:, 1:]
            if self.mode == 'train':
                num_data = np.delete(num_data, self._data_interval(num_data.shape[0]), 0)
            if self.mode == 'val':
                num_data = num_data[self._data_interval(num_data.shape[0]), :]

            labels = num_data[:, -1:].reshape(-1)
            self.labels = torch.from_numpy(self.label_encoder.transform(labels))
            num_data = num_data[:, :-1].astype(int)

        else:
            num_data = self.np_data[:, 1:].astype(int)

        #str_data = self.np_data[:, self.str_idx]
        num_data = num_data / np.max(num_data, axis=0)  # normalization
        self.num_data = torch.from_numpy(num_data).type(torch.FloatTensor)
        # self.str_data = torch.from_numpy(str_data)[:, :]

    def get_encoder(self):
        return self.label_encoder

    def grouping(self):
        group = []
        num = 0
        prev_n = 1
        for i, name in enumerate(self.feature_name):
            if i == 0 or i in self.str_idx:
                continue
            if name == 'knowcode':
                continue
            n = int(name.split('_')[0][2:])
            if n != prev_n:
                group.append(num)
                num = 0
            num += 1
            prev_n = n
        group.append(num)
        return group

    def __len__(self):
        return self.num_data.shape[0]

    def __getitem__(self, item):
        if self.mode == 'test':
            return self.num_data[item]
        else:
            return self.num_data[item], self.labels[item]


def create_data_loaders(year, args, forward=False):
    if forward:
        mode = 'test'
        dataset = KnowData(year, mode, args)
        dataloader = DataLoader(
            dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True,
        )
        return dataloader

    else:
        '''
            if forward is False and args['FOLD'] == K,
                it will return K (train, val) dataloader tuples in list
        '''
        dataset = []
        data_list = []
        for k in range(args['FOLD']):
            train_dataset = KnowData(year, 'train', args, k + 1)
            val_dataset = KnowData(year, 'val', args, k + 1)

            train_loader = DataLoader(
                train_dataset, batch_size=args['batch_size'], shuffle=True,
                num_workers=args['num_workers'], pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args['batch_size'], shuffle=True,
                num_workers=args['num_workers'], pin_memory=True,
            )
            dataset.append((train_dataset, val_dataset))
            data_list.append((train_loader, val_loader))

            label_encoder = train_dataset.get_encoder()

        return data_list, dataset, label_encoder


def create_exp_loaders(year, args, fold, is_train, batch_size):
    if is_train:
        train_dataset = KnowData(year, 'train', args, fold)
        train_loader = DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True,
            num_workers=args['num_workers'], pin_memory=True,
        )
        return train_loader
    else:
        val_dataset = KnowData(year, 'val', args, fold)
        val_loader = DataLoader(
            val_dataset, batch_size=args['batch_size'], shuffle=True,
            num_workers=args['num_workers'], pin_memory=True,
        )
        return val_loader

if __name__ == '__main__':
    from pathlib import Path

    year = 2017

    args = {}
    args['FOLD'] = 5
    args['DATA_ROOT'] = Path('../KNOW_data/')
    args['STRING_INDEX'] = [87, 88, 89, 92, 121, 139, 140, 141, 142, 143, 148]
    args['batch_size'] = 4
    args['num_workers'] = 1

    data_list = []

    label_encoder_list = []

    for k in range(args['FOLD']):
        train_dataset = KnowData(year, 'train', args, k + 1)
        val_dataset = KnowData(year, 'val', args, k + 1)

        train_loader = DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True,
            num_workers=args['num_workers'], pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args['batch_size'], shuffle=True,
            num_workers=args['num_workers'], pin_memory=True,
        )
        data_list.append((train_loader, val_loader))

        label_encoder = train_dataset.get_encoder()
        group = train_dataset.grouping()
