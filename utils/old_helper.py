import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import os
import random
import datetime
from pytz import timezone

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import tensorflow.keras as keras


##------------------------------------------------------------Importing the Dataset------------------------------------------------------------##


def processing_folder(folder_path, sample_size, train_size):
    # os.chdir(folder_path)
    path = os.path.join(os.getcwd(), folder_path)
    files = sorted(os.listdir(folder_path), key=lambda x: int(re.findall(r'\d+', x)[0]))
    train_samples = []
    test_samples = []
    samples = []
    cols = []
    for i, path in enumerate(files):
        path = os.path.join(os.getcwd(), folder_path, path)
        if i >= sample_size:
            break
        with open(path, encoding='utf-8',           ##https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
                 errors='ignore') as f:
            lines = f.readlines()
        ## extract column names (once)
        if i == 0:
            ls = lines[0].split('\t')
            if re.findall(r'\w+|\d+', ls[-1]):
                ls[-1] = re.findall(r'\w+|\d+', ls[-1])[0]
                cols = ls
        ## extracting all samples from the current file
        sample = []
        curr_text_id = ''
        curr_index = -1
        for line in lines[1:]:
            ls = line.split('\t')
            # if len(ls) != 9:
            #     print(path, line)
            if re.findall(r'\w+|\d+', ls[-1]):
                ls[-1] = re.findall(r'\w+|\d+', ls[-1])[0]
                if ls[1] != curr_text_id:
                    curr_index = 0
                    curr_text_id = ls[1]
                else:
                    curr_index += 1
                ls.append(curr_index)
                sample.append(ls)
        ##  split the current data into train-test-sets
        split_index = int(train_size * len(sample))
        train_samples = train_samples + sample[:split_index]
        test_samples = test_samples + sample[split_index:]
        samples = samples + sample
    ## forming dataframes
    df_all = pd.DataFrame(samples)
    df_train = pd.DataFrame(train_samples)
    df_test = pd.DataFrame(test_samples)
    ## renaming columns
    cols = cols + ['INDEX']
    df_all.columns, df_train.columns, df_test.columns = cols, cols, cols
    ## construct onehot encoders
    df_all['K1'], df_all['K2'] = df_all['KEYCODE'], df_all['KEYCODE']
    uni_encoder = OneHotEncoder().fit(df_all[['KEYCODE']])
    di_encoder = OneHotEncoder().fit(df_all[['K1', 'K2']]) 
    return df_all, df_train, df_test, uni_encoder, di_encoder