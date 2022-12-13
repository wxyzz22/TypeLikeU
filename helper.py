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

def processing_folder(folder_path, sample_size, train_size, dev_size):
    '''
    Import and process the keystroke data folder conating .txt files of users' keytroke data
    '''
    # os.chdir(folder_path)
    path = os.path.join(os.getcwd(), folder_path)
    files = sorted(os.listdir(path), key=lambda x: int(re.findall(r'\d+', x)[0]))
    train_samples = []
    dev_samples = []
    test_samples = []
    samples = []
    cols = []
    # text_count_id = 0
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
            # ## debugdding block (to detect misaligned rows in files)
            # if len(ls) != 9:
            #     print(path, line)
            if re.findall(r'\w+|\d+', ls[-1]):
                ls[-1] = re.findall(r'\w+|\d+', ls[-1])[0]
                if ls[1] != curr_text_id:
                    curr_index = 0
                    curr_text_id = ls[1]
                    # text_count_id += 1
                else:
                    curr_index += 1
                # ls.extend([curr_index, text_count_id])
                ls.append(curr_index)
                sample.append(ls)
        ##  split the current data into train-test-sets
        split_index_1 = int(train_size * len(sample))
        split_index_2 = split_index_1 + int(dev_size * len(sample))
        train_samples = train_samples + sample[:split_index_1]
        dev_samples = dev_samples + sample[split_index_1:split_index_2]
        test_samples = test_samples + sample[split_index_2:]
        samples = samples + sample
    ## forming dataframes: total data, train, test, dev
    data = pd.DataFrame(samples)
    train_data = pd.DataFrame(train_samples)
    dev_data = pd.DataFrame(dev_samples)
    test_data = pd.DataFrame(test_samples)
    ## renaming columns
    # cols = cols + ['INDEX'] + ['TEXT_COUNT_ID']
    cols = cols + ['INDEX']
    data.columns, train_data.columns, dev_data.columns, test_data.columns = cols, cols, cols, cols
    ## construct onehot encoders from train data
    train_data['K1'], train_data['K2'] = train_data['KEYCODE'], train_data['KEYCODE']
    KEYCODE_enc = OneHotEncoder(handle_unknown='ignore').fit(train_data[['KEYCODE']])
    K1_enc = OneHotEncoder(handle_unknown='ignore').fit(train_data[['K1']])
    K2_enc = OneHotEncoder(handle_unknown='ignore').fit(train_data[['K2']])
    K1_K2_enc = OneHotEncoder(handle_unknown='ignore').fit(train_data[['K1', 'K2']])
    train_data = train_data.drop(columns=['K1', 'K2'])
    return data, train_data, dev_data, test_data, KEYCODE_enc, K1_K2_enc, K1_enc, K2_enc


def processing_meta(file_path):
    '''
    Import and process the `metadata_participants.txt` file.
    '''
    path = os.path.join(os.getcwd(), file_path)
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    ls = lines[0].split('\t')
    if re.findall(r'\w+|\d+', ls[-1]):
        ls[-1] = re.findall(r'\w+|\d+', ls[-1])[0]
        cols = ls
    meta = []
    for line in lines[1:]:
        ls = line.split('\t')
        if re.findall(r'\w+|\d+', ls[-1]):
            ror_rate = re.findall(r'\w+|\d+', ls[-1])[0]
            for char in re.findall(r'\w+|\d+', ls[-1])[1:]:
                ror_rate = ror_rate + '.' + char
            ls[-1] = ror_rate
        meta.append(ls)
    meta = pd.DataFrame(meta)
    ## changing column names
    meta.columns = cols
    ## correcting dtypes of columns
    meta[['AGE', 'TIME_SPENT_TYPING']] = meta[['AGE', 'TIME_SPENT_TYPING']].astype(int)
    meta[['ERROR_RATE', 'AVG_WPM_15', 'AVG_IKI', 'ECPC', 'KSPC', 'ROR']] = meta[['ERROR_RATE', 'AVG_WPM_15', 'AVG_IKI', 'ECPC', 'KSPC', 'ROR']].astype(float)
    meta[['HAS_TAKEN_TYPING_COURSE']] = meta[['HAS_TAKEN_TYPING_COURSE']].astype(bool)
    ## processing FINGERS into integers (ordinal)
    mask = {}
    mask[1] = meta['FINGERS'] == '1-2'
    mask[3] = meta['FINGERS'] == '3-4'
    mask[5] = meta['FINGERS'] == '5-6'
    mask[7] = meta['FINGERS'] == '7-8'
    mask[9] = meta['FINGERS'] == '9-10'
    mask[10] = meta['FINGERS'] == '10+'
    for val in mask:
        meta.loc[mask[val], 'FINGERS'] = val
    return meta



##------------------------------------------------------------Keyboard Layout Encoding------------------------------------------------------------##

def get_qwerty_keyboard():
    '''
    Return the encoded keyboard into javascript keycodes as pandas dataframe
    '''
    first_row = [27, 27, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 0, 0, 145, 126, 0, 0, 0, 0, 0]
    space = [0] * 23
    second_row = [192, 49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 189, 187, 8, 0, 45, 36, 33, 0, 144, 111, 106, 109]
    third_row = [9, 81, 87, 69, 82, 84, 89, 85, 73, 79, 80, 219, 221, 220, 0, 46, 35, 34, 0, 103, 104, 105, 107]
    fourth_row = [20, 65, 83, 68, 70, 71, 72, 74, 75, 76, 186, 222, 13, 13, 0, 0, 0, 0, 0, 100, 101, 102, 107]
    fifth_row = [16, 16, 90, 88, 67, 86, 66, 78, 77, 188, 190, 191, 16, 16, 0, 0, 38, 0, 0, 97, 98, 99, 13]
    sixth_row = [17, 17, 191, 18, 32, 32, 32, 32, 32, 18, 92, 93, 17, 17, 0, 37, 40, 39, 0, 96, 96, 110, 13]
    qwerty_keyboard = pd.DataFrame({'1st': first_row,
                                    'space': space,
                                    '2nd': second_row,
                                    '3rd': third_row,
                                    '4th': fourth_row,
                                    '5th': fifth_row,
                                    '6th': sixth_row}).transpose()
    qwerty_keyboard.index = list(range(7))
    return qwerty_keyboard

class Keyboard:
    def __init__(self, keyboard_df=get_qwerty_keyboard()):
        self.keyboard = keyboard_df
        self.keycode_pos = self.get_keycode_pos()
    
    def get_keycode_pos(self):
        '''
        Generates Python dictionary encoding the keyboard keycode positions, i.e.
              - keys = javascript keycode
              - values = [i, j] of the corresponding keycode position on the keyboard
        Return: Python dict
        '''
        keyboard_dict = {}
        for row in self.keyboard.index:
            for col, entry in enumerate(self.keyboard.iloc[row, :]):
                if entry in keyboard_dict:
                    keyboard_dict[entry].append([row, col])
                else:
                    keyboard_dict[entry] = [[row, col]]
        return keyboard_dict
        
    def keycode_distance(self, keycode1, keycode2):
        '''
        Given a pair of keycodes, return their relative distance on the keyboard
        '''
        keycode1 = int(keycode1)
        keycode2 = int(keycode2)
        def manhattan_dist(arr1, arr2):
            return abs(arr1[0] - arr2[0]) + abs(arr1[1] - arr2[1])
        distance = 30 ## any integer larger than 22+6
        if keycode1 in self.keycode_pos and keycode2 in self.keycode_pos:
            for arr1 in self.keycode_pos[keycode1]:
                for arr2 in self.keycode_pos[keycode2]:
                    curr_dist = manhattan_dist(arr1, arr2)
                    if curr_dist < distance:
                        distance = curr_dist
            if distance < 5:
                return distance
        return 5
    
    def home_distance(self, keycode_list):
        '''
        Computes the AVERAGE distance of a list of keycodes to the home keys, where
        In QWERTY keyboard, F and J are the home keys with keycodes 70 and 74 resp.
        '''
        sum = 0
        for key in keycode_list:
            key = int(key)
            sum += min([self.keycode_distance(70, key), self.keycode_distance(74, key)])
        return sum/len(keycode_list)
    
    def keyboard_dict(self):
        return {'keycode': self.keycode_distance, 'home': self.home_distance}
    
    
    
##------------------------------------------------------------Extracting Features------------------------------------------------------------##

class Extractor:
    def __init__(self, sub_data, keyboard_dict=Keyboard().keyboard_dict(), latencies=['HL', 'PL', 'IL', 'RL']):
        self.keyboard_dict = keyboard_dict
        self.latencies = latencies

        self.unigraph = self.unigraph_extractor(sub_data)
        self.digraph = self.digraph_extractor(sub_data)
    
    def unigraph_extractor(self, df, user_str=True, keycode_str=True, drop_user=False):
        '''
        Generates unigraph related features and returns the dataframe
        '''
        df = df[['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'INDEX']]
        df = df.astype('float64')
        if user_str:
            df['PARTICIPANT_ID'] = df['PARTICIPANT_ID'].astype('int64').astype(str)
        df = df.rename(columns={'PARTICIPANT_ID': 'USER'})
        if drop_user:
            df = df.drop(columns=['USER'])
        if keycode_str:
            df['KEYCODE'] = df['KEYCODE'].astype('int64').astype(str)
        ## construct new features
        if 'HL' in self.latencies:
            df['HL'] = df['RELEASE_TIME'] - df['PRESS_TIME']
        if 'IL' in self.latencies:
            df['IL'] = pd.concat([df['PRESS_TIME'][1:], pd.Series([0])], ignore_index=True) - df['RELEASE_TIME']
        if 'RL' in self.latencies:
            df['RL'] = pd.concat([df['RELEASE_TIME'][1:], pd.Series([0])], ignore_index=True) - df['RELEASE_TIME']
        if 'PL' in self.latencies:
            df['PL'] = pd.concat([df['PRESS_TIME'][1:], pd.Series([0])], ignore_index=True) - df['PRESS_TIME']
        ## dropping rows where the NEXT row has INDEX==0 (indicating a transition to next sentence)
        shift_txt = pd.concat([df['TEST_SECTION_ID'][1:], df['TEST_SECTION_ID'][-1:]], ignore_index=True) - df['TEST_SECTION_ID']
        mask = shift_txt == 0
        df = df.loc[mask]
        ## cleaning irrelavant info
        df = df.drop(columns=['PRESS_TIME', 'RELEASE_TIME', 'TEST_SECTION_ID'])
        df = df.iloc[:-1, :]
        return df
    
    def digraph_extractor(self, df, user_str=True, keycode_str=True, drop_user=False):
        '''
        Generates digraph related features and returns the dataframe
        '''
        df = df[['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'INDEX']]
        df = df.astype('float64')
        if user_str:
            df['PARTICIPANT_ID'] = df['PARTICIPANT_ID'].astype('int64').astype(str)
        df = df.rename(columns={'PARTICIPANT_ID': 'USER'})
        if drop_user:
            df = df.drop(columns=['USER'])
        ## construct new features
        df['K1'] = df['KEYCODE']
        df['K2'] = pd.concat([df['KEYCODE'][1:], pd.Series([0])], ignore_index=True)
        if keycode_str:
            df['K1'] = df['K1'].astype('int64').astype(str)
            df['K2'] = df['K2'].astype('int64').astype(str)
        df['I1'] = df['INDEX']
        df['I2'] = pd.concat([df['INDEX'][1:], pd.Series([0])], ignore_index=True)
        if 'HL' in self.latencies:
            df['HL1'] = df['RELEASE_TIME'] - df['PRESS_TIME']
            df['HL2'] = pd.concat([df['HL1'][1:], pd.Series([0])], ignore_index=True)
        if 'IL' in self.latencies:
            df['IL'] = pd.concat([df['PRESS_TIME'][1:], pd.Series([0])], ignore_index=True) - df['RELEASE_TIME']
        if 'RL' in self.latencies:
            df['RL'] = pd.concat([df['RELEASE_TIME'][1:], pd.Series([0])], ignore_index=True) - df['RELEASE_TIME']
        if 'PL' in self.latencies:
            df['PL'] = pd.concat([df['PRESS_TIME'][1:], pd.Series([0])], ignore_index=True) - df['PRESS_TIME']
        ## dropping instances where I2 is zero (indicating a transition to next sentence)
        shift_txt = pd.concat([df['TEST_SECTION_ID'][1:], df['TEST_SECTION_ID'][-1:]], ignore_index=True) - df['TEST_SECTION_ID']
        mask = shift_txt == 0
        df = df.loc[mask]
        ## cleaning irrelavant info
        df = df.drop(columns=['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'INDEX', 'TEST_SECTION_ID'])
        df = df.iloc[:-1, :]
        return df  
    
    def unigraph_avg(self, avg_mode, data=None, drop_origin=True, rename_avg=True, round_avg=True):
        '''
        Generates unigraph with values replaced by average behavior
        Input:
            avg_mode: str, takes value in 'mean', 'median' (if not == 'mean', default to 'median')
        Output:
            unigraph dataframe with Average User data
        '''
        if data:
            df = data.copy()
        else:
            df = self.unigraph.copy()
        for XL in self.latencies:
            df[XL+'_avg'] = df[XL]
        for keycode in df['KEYCODE'].unique():
            mask = df['KEYCODE'] == keycode
            if avg_mode == 'mean':
                avg_df = df.loc[mask, self.latencies].mean()
            else:
                avg_df = df.loc[mask, self.latencies].median()
            for XL in self.latencies:
                df.loc[mask, XL+'_avg'] = avg_df[XL]
        if round_avg:
            for XL in self.latencies:
                df[XL+'_avg'] = round(df[XL+'_avg'])
        if drop_origin:
            df = df.drop(columns=self.latencies)
        if drop_origin and rename_avg:
            df = df.rename(columns=lambda name: name[:2] if '_avg' in name else name)
        return df
    
    ## https://towardsdatascience.com/do-you-use-apply-in-pandas-there-is-a-600x-faster-way-d2497facfa66
    def digraph_avg(self, avg_mode, data=None, drop_origin=True, rename_avg=True, round_avg=True):
        '''
        Generates digraph with values replaced by average behavior
        Input:
            avg_mode: str, takes value in 'mean', 'median' (if not == 'mean', default to 'median')
        Output:
            digraph dataframe with Average User data
        '''
        if data:
            df = data.copy()
        else:
            df = self.digraph.copy()
        df['K1_K2'] = df[['K1', 'K2']].apply(tuple, axis=1)
        latencies = self.latencies.copy()
        if 'HL' in latencies:
            latencies.remove('HL')
            latencies.insert(0, 'HL2')
            latencies.insert(0, 'HL1')
        for XL in latencies:
            df[XL+'_avg'] = df[XL]
        for pair in df['K1_K2'].unique():
            mask = df['K1_K2'] == pair
            if avg_mode == 'mean':
                avg_df = df.loc[mask, latencies].mean()
            else:
                avg_df = df.loc[mask, latencies].median()
            for XL in latencies:
                df.loc[mask, XL+'_avg'] = avg_df[XL]
        if round_avg:
            for XL in latencies:
                df[XL+'_avg'] = round(df[XL+'_avg'])
        if drop_origin:
            df = df.drop(columns=latencies+['K1_K2'])
        if drop_origin and rename_avg:
            df = df.rename(columns=lambda name: re.search(r'(.{2,3})(_avg)', name).group(1) if '_avg' in name else name)
        return df
 
    def unigraph_keyboard(self, avg_mode=None):
        '''
        Returns a unigraph dataframe with added keyboard layout features
        '''
        if avg_mode:
            df = self.unigraph_avg(avg_mode)
        else:
            df = self.unigraph.copy()
        home_dist = []
        for row in df.index:
            home_dist.append(self.keyboard_dict['home']([df['KEYCODE'][row]]))
        df['HD'] = home_dist
        # cols = list(df.columns[:-3]) + list(df.columns[-1:]) + list(df.columns[-3:-1])
        num_cols = len(df.columns)
        cols = list(df.columns[:num_cols-1-len(self.latencies)]) + list(df.columns[-1:]) + list(df.columns[-1-len(self.latencies):-1])
        df = df[cols]
        return df
    
    def digraph_keyboard(self, avg_mode=None):
        '''
        Returns a digraph dataframe with added keyboard layout features
        '''
        if avg_mode:
            df = self.digraph_avg(avg_mode)
        else:
            df = self.digraph.copy()
        keycode_dist = []
        home_dist = []
        for row in df.index:
            keycode_dist.append(self.keyboard_dict['keycode'](df['K1'][row], df['K2'][row]))
            home_dist.append(self.keyboard_dict['home']([df['K1'][row], df['K2'][row]]))
        df['KD'] = keycode_dist
        df['HD'] = home_dist
        # cols = list(df.columns[:-5]) + list(df.columns[-2:]) + list(df.columns[-5:-2])
        num_cols = len(df.columns)
        cols = list(df.columns[:num_cols-2-(len(self.latencies)+1)]) + list(df.columns[-2:]) + list(df.columns[-2-(len(self.latencies)+1):-2])
        df = df[cols]
        return df
    
    def IQR_filter(self, data, fold):
        Q3 = data.quantile(.75)
        Q1 = data.quantile(.25)
        IQR = Q3 - Q1
        max = Q3 + fold * IQR
        min = Q1 - fold * IQR
        return min, max

    def ABS_filter(self, data, bounds):
        num_bottom, num_top = bounds
        min = data.sort_values()[:num_bottom+1].values[-1]
        max = data.sort_values(ascending=False)[:num_top+1].values[-1]
        return min, max
    
    def unigraph_filtered(self, avg_mode, encode_keyboard, outlier_filter, bounds_dict):
        '''
        Input:
            avg_mode: str, takes value in ['median', 'mean', None(default)]
            encode_keyboard: boolean
            filter: str, takes value in ['ABS'(default), 'IQR']
            bounds_dict: a python dictionary with keys=latencies, 
                                                  values=needed params
                      ==> for IQR: values = folds (i.e. scaling IQR by fold*IQR)
                      ==> for ABS: values = [num_bottoms, num_tops]
        '''
        filter_latencies = list(bounds_dict.keys())
        if encode_keyboard:
            df = self.unigraph_keyboard(avg_mode)
        elif avg_mode:
            df = self.unigraph_avg(avg_mode)
        else:
            df = self.unigraph.copy()
        for latency in filter_latencies:
            for user in df['USER'].unique():
                mask_user = df['USER'] == user
                mask_nonuser = df['USER'] != user
                subdf = df.loc[mask_user, latency]
                if outlier_filter == 'IQR':
                    min, max = self.IQR_filter(subdf, bounds_dict[latency])
                else:
                    min, max = self.ABS_filter(subdf, bounds_dict[latency])
                mask_max = df[latency] <= max
                mask_min = df[latency] >= min
                df = df.loc[mask_user & mask_max & mask_min | mask_nonuser]
        return df
    
    def digraph_filtered(self, avg_mode, encode_keyboard, outlier_filter, bounds_dict):
        '''
        Input:
            avg_mode: str, takes value in ['median', 'mean', None(default)]
            encode_keyboard: boolean
            filter: str, takes value in ['ABS'(default), 'IQR']
            bounds_dict: a python dictionary with keys=latencies, 
                                                  values=needed params
                      ==> for IQR: values = folds (i.e. scaling IQR by fold*IQR)
                      ==> for ABS: values = [num_bottoms, num_tops]
        '''
        filter_latencies = list(bounds_dict.keys())
        if encode_keyboard:
            df = self.digraph_keyboard(avg_mode)
        elif avg_mode:
            df = self.digraph_avg(avg_mode)
        else:
            df = self.digraph.copy()
        for latency in filter_latencies:
            for user in df['USER'].unique():
                mask_user = df['USER'] == user
                mask_nonuser = df['USER'] != user
                subdf = df.loc[mask_user, latency]
                if outlier_filter == 'IQR':
                    min, max = self.IQR_filter(subdf, bounds_dict[latency])
                else:
                    min, max = self.ABS_filter(subdf, bounds_dict[latency])
                mask_max = df[latency] <= max
                mask_min = df[latency] >= min
                df = df.loc[mask_user & mask_max & mask_min | mask_nonuser]
        return df

    

##------------------------------------------------------------KDS Object (Sequential Input)------------------------------------------------------------##

class KDS:
    def __init__(self, df, 
                 n_steps, shift, batch_size, 
                 nonkeycodeB_features, output_features,
                 encoders, enc_names, do_onehot=True):
        self.df = df
        self.window_length = n_steps + 1
        self.n_steps = n_steps
        self.shift = shift
        self.batch = batch_size

        self.inputB_features = nonkeycodeB_features        ## inputB_features are the features input to keycode_embed layer, not including keycodes
        self.output_features = output_features
        
        for i, user in enumerate(self.df['USER'].unique()):
            mask = self.df['USER'] == user
            user_df = self.df.loc[mask, :]
            ## One-hot on 'KEYCODE' (mode=='uni') OR 'K1', 'K2' (mode!='uni')
            if len(encoders) == 1 and enc_names[0] == 'KEYCODE':
                keycode_np = encoders[0].transform(user_df[['KEYCODE']].astype(str)).toarray()
            elif len(encoders) == 1 and enc_names[0] == 'K1_K2':
                keycode_np = encoders[0].transform(user_df[['K1', 'K2']].astype(str)).toarray()
            else:
                k1_onehot = encoders[0].transform(user_df[['K1']].astype(str)).toarray()
                k2_onehot = encoders[1].transform(user_df[['K2']].astype(str)).toarray()
                keycode_np = k1_onehot + k2_onehot
            curr_df = np.concatenate([keycode_np, self.df.loc[mask, self.inputB_features+self.output_features]], axis=1)
            ## get the TFDS dataset of inputs (inputA, inputB) and output
            curr_in, curr_out = self.get_dataset(curr_df)
            if i == 0:
                self.ds_in = curr_in
                self.ds_out = curr_out
            else:
                self.ds_in = self.ds_in.concatenate(curr_in)
                self.ds_out = self.ds_out.concatenate(curr_out)
        ## zip the TFDS inputs and output for easy access at training
        self.ds = tf.data.Dataset.zip((self.ds_in, self.ds_out))
        
        for inputA, inputB in self.ds_in.take(1):
            self.inputA = inputA.shape
            self.inputB = inputB.shape
        
        for output in self.ds_out.take(1):
            self.output = output.shape

    def get_dataset(self, df):
        dataset = tf.data.Dataset.from_tensor_slices(df).window(size=self.window_length, shift=self.shift, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_length)).batch(self.batch)
        ds_in = dataset.map(lambda window: (window[:, :self.n_steps, :], window[:, -1, :-len(self.output_features)]))
        ds_in = ds_in.prefetch(tf.data.AUTOTUNE)
        ds_out = dataset.map(lambda window: window[:, -1, -len(self.output_features):])
        ds_out = ds_out.prefetch(tf.data.AUTOTUNE)
        return ds_in, ds_out


##------------------------------------------------------------KDI Object (Image-like Input)------------------------------------------------------------##

class KDI:
    def __init__(self, train_data, df, 
                 n_steps, shift, batch_size, mat_length, 
                 inputA_features, inputB_features, output_features, 
                 inputB_type='image', encoders=None, keep_smaller_window=False, add_UNK=True):
        self.train_data = train_data
        self.df = df
        self.n_steps = n_steps
        self.shift = shift
        self.batch_size = batch_size
        self.mat_length = mat_length

        self.inputA_features = inputA_features
        self.inputB_features = inputB_features
        self.inputB_type = inputB_type                ## 'label, 'onehot', or 'image', else default to 'image'
        self.output_features = output_features        ## output_features ('HL', 'XL' in ['PL', 'RL', 'IL'])

        self.encoders = encoders
        self.keep_smaller_window = keep_smaller_window
        self.add_UNK = add_UNK

        self.keycode_dict = self.keycode_topfreq_dict(top=self.mat_length-1)

        self.inputA, self.inputB, self.output = self.kdi_training_data()
        self.ds = self.generate_kdi()


    def keycode_topfreq_dict(self, top):
        '''
        generate dictionary for the most popular `top` many keycodes using training data
        '''
        keycode_dict = {keycode: i for i, keycode in enumerate(self.train_data['KEYCODE'].astype('int32').value_counts()[:top].to_dict().keys())}
        if self.add_UNK:
            keycode_dict[0] = len(keycode_dict)
        return keycode_dict
  

    def single_input_image(self, curr_chunk, features, mat_length, keycode_dict):
        '''
        Helper function to generate a single image (with # of color channels == # of features)
        '''
        mat_dict = {}
        for feature in features:
            mat_dict['mat_'+feature] = np.zeros((mat_length, mat_length))
        mat_dict['count'] = np.zeros((mat_length, mat_length))

        for row in curr_chunk.index:
            i = int(curr_chunk.loc[row, 'K1'])
            j = int(curr_chunk.loc[row, 'K2'])
            if i in keycode_dict:
                pos_i = keycode_dict[i]
            else:
                pos_i = keycode_dict[0]   ## pos_i = top (the last key-value pair)
            if j in keycode_dict:
                pos_j = keycode_dict[j]
            else:
                pos_j = keycode_dict[0]
            for feature in features:
                if feature != 'HL':
                    mat_dict['mat_'+feature][pos_i, pos_j] += curr_chunk.loc[row, feature]
                else:
                    mat_dict['mat_'+feature][pos_i, pos_j] += (i + j) / 2
            mat_dict['count'][pos_i, pos_j] += 1
        mask_nonzero = mat_dict['count'] != 0
        mat_ls = []
        for feature in features:
            mat_dict['mat_'+feature][mask_nonzero] = mat_dict['mat_'+feature][mask_nonzero] / mat_dict['count'][mask_nonzero]
            mat_ls.append(mat_dict['mat_'+feature])
        return np.stack(mat_ls, axis=-1)
    

    def single_kdi_input(self, curr_chunk):
        '''
        Generates the group of inputA, inputB, and Output of the current chunk
        '''
        last_index = curr_chunk.index[-1]
        output_ls = []
        for feature in self.output_features:
            output_ls.append(curr_chunk.loc[last_index, feature])
        output_np = np.array(output_ls)
        ## inputA 
        inputA = self.single_input_image(curr_chunk.iloc[:-1], self.inputA_features, self.mat_length, self.keycode_dict)
        ## inputB
        if self.inputB_type == 'label':
            inputB = np.array(curr_chunk.loc[last_index, self.inputB_features + ['K1', 'K2']])
        elif self.inputB_type == 'onehot' and self.encoders:
            if len(self.encoders) == 1:
                inputB_keycode = self.encoders[0].transform(curr_chunk.loc[[last_index], ['K1', 'K2']].astype(str)).toarray()
            else:
                inputB_k1 = self.encoders[0].transform(curr_chunk.loc[[last_index], ['K1']].astype(str)).toarray()
                inputB_k2 = self.encoders[1].transform(curr_chunk.loc[[last_index], ['K2']].astype(str)).toarray()
                inputB_keycode = inputB_k1 + inputB_k2
            inputB = np.concatenate([inputB_keycode, np.array(curr_chunk.loc[last_index, self.inputB_features])], axis=1)
        else:
            inputB = self.single_input_image(curr_chunk.iloc[-1:], self.inputB_features, self.mat_length, self.keycode_dict)
        return inputA, inputB, output_np
    

    def kdi_training_data(self):
        '''
        Generates numpy arrays of inputA=(total_images, mat_length, mat_length, # of features), inputB, output
        '''
        window_length = self.n_steps + 1
        inputA_arr, inputB_arr, output_arr = [], [], []
        for user in self.df['USER'].unique():
            curr_df = self.df[self.df['USER'] == user]
            i = 0
            while i+window_length < len(curr_df):
                curr_chunk = curr_df.iloc[i:i+window_length]
                curr_inputA, curr_inputB, curr_output = self.single_kdi_input(curr_chunk)
                inputA_arr.append(curr_inputA)
                inputB_arr.append(curr_inputB)
                output_arr.append(curr_output)
                i += self.shift
            if self.keep_smaller_window and i < len(curr_df) - 1:    ## i cannot be curr_df[-1:] of length 1, since impossible to split into input and output data
                curr_chunk = curr_df.iloc[i:]
                curr_inputA, curr_inputB, curr_output = self.single_kdi_input(curr_chunk)
                inputA_arr.append(curr_inputA)
                inputB_arr.append(curr_inputB)
                output_arr.append(curr_output)
        return np.stack(inputA_arr, axis=0), np.stack(inputB_arr, axis=0), np.stack(output_arr, axis=0)
    

    def generate_kdi(self):
        '''
        Prepared tf.data object for training (batched)
        '''
        dataset = tf.data.Dataset.from_tensor_slices(({'inputA': self.inputA, 'inputB': self.inputB}, 
                                                      self.output)).batch(self.batch_size)
        return dataset
    

    

##------------------------------------------------------------Functionalized Callbacks------------------------------------------------------------##

def create_checkpoint_callback(experiment_name, 
                               avg_mode,
                               save_weights_only=True, 
                               monitor='val_loss', 
                               mode='min', 
                               save_best_only=True,
                               path='TypeLikeU-COMP576-FinalProject/exp_records',
                               time_zone='America/Chicago'):
    now_time = datetime.datetime.now(timezone(time_zone))
    checkpoint_filepath = path + "/" + "checkpoints" + "/" + experiment_name + "/" + now_time.strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = checkpoint_filepath + '-avg' if avg_mode else checkpoint_filepath
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                             save_weights_only=save_weights_only,
                                                             monitor=monitor,
                                                             mode=mode,
                                                             save_best_only=save_best_only)
    print(f"Saving ModelCheckpoint files to :{checkpoint_filepath}")
    return checkpoint_callback

def create_lr_scheduler(max_cap):
    def lr_finder(epoch):
        num1 = 4 - (epoch - 1) // 3
        num2 = 1 + (epoch - 1) % 3 * 3
        lr = round(0.1 ** num1 * num2, 7)
        if max_cap and lr > max_cap:
            return max_cap
        return lr
    return tf.keras.callbacks.LearningRateScheduler(lr_finder)

def create_tensorboard_callback(experiment_name, 
                                avg_mode, 
                                path='TypeLikeU-COMP576-FinalProject/exp_records', 
                                time_zone='America/Chicago'):
    now_time = datetime.datetime.now(timezone(time_zone))
    log_dir = path + "/" + "tensorboard" + "/" + experiment_name + "/" + now_time.strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir + '-avg' if avg_mode else log_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to :{log_dir}")
    return tensorboard_callback

def create_earlystopping_callback(patience, monitor='val_loss'):
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)

def get_callbacks(experiment_name, patience, avg_mode):
    '''
    Returns the callbacks list consists of EarlyStopping, ModelCheckpoint, TensorBoard
    '''
    earlystopping = create_earlystopping_callback(patience)
    modelcheckpoint = create_checkpoint_callback(experiment_name=experiment_name, avg_mode=avg_mode)
    tensorboard = create_tensorboard_callback(experiment_name=experiment_name, avg_mode=avg_mode)
    return [earlystopping, modelcheckpoint, tensorboard]



##------------------------------------------------------------Miscellaneous Functions------------------------------------------------------------##
def lr_vs_loss(history):
    '''
    Generates the plot of learning rates versus loss.
    Input:
        history: a model history object containing 'lr'
    '''
    lrs = history.history['lr']
    loss = history.history['val_loss']
    plt.semilogx(lrs, loss);


def random_boxplot(df, latency='PL', width=6):
    fig, ax = plt.subplots(width, width, figsize=(width*3.5, width*10))
    user_ls = random.sample(list(df['USER'].unique()), width**2)
    for i, user in enumerate(user_ls):
        mask = df['USER'] == user
        row = i // width
        col = i % width
        df.loc[mask, latency].plot.box(ax=ax[row, col])
        ax[row, col].set_xlabel(f'User: {user}')
    

##------------------------------------------------------------Check Import Successful------------------------------------------------------------##
print("Hello from helper.py")