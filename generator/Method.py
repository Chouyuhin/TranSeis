#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, Activation, LSTM, Conv1D, InputSpec
from tensorflow.keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class DataGenerator(keras.utils.Sequence):
    
    """ 
    
    list_IDsx: str
        List of trace names.
            
    input_dir: str
        dir of hdf5 file containing waveforms data and csv gile containing metadata.
        args['input_dir'] + 'DiTing330km_part_{}.hdf5'.format(part)

    num_part: int
        number of dataset chunk 
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    phase_window: int, fixed=40
        The number of samples (window) around each phaset.
            
    shuffle: bool, default=True
        Shuffeling the list.
            

    coda_ratio: {float, 0.4}, default=0.4
        % of S-P time to extend event/coda envelope past S pick.       
            
    shift_event_r: {float, None}, default=0.9
        Rate of randomly shifting the event within a trace. 
            

    Returns
    --------        
    Batches of two dictionaries: 
    {'input': X}: pre-processed waveform as input 
    {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 list_IDs,
                 input_dir,
                 num_part,
                 dim, 
                 batch_size=50, 
                 n_channels=3, 
                 phase_window= 40, 
                 shuffle=True,                 
                 coda_ratio = 0.4,
                 shift_event_r = None):
       
        'Initialization'
        self.input_dir = input_dir
        self.num_part = num_part
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs     
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()      
        self.coda_ratio = coda_ratio
        self.shift_event_r = shift_event_r



    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        list_IDs_temp = []
        
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]  
        chunk_size =  len(self.list_IDs)//self.num_part     # 60000
        for i in range(0, len(self.list_IDs), chunk_size):
            chunk = self.list_IDs[i:i+chunk_size]      # 60000   chunk[0] = 031999.0119
            list_IDs_temp.append([chunk[j] for j in range(self.batch_size)])     # whole batch of IDs from 27 files
        # list_IDs_temp： [['031999.0119', '016171.0374'], ['053929.0007', '052614.0404']]    每个part的batch，这里是2       
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]    # batch of IDs
        
        X, y1, y2, y3 = self.__data_generation(list_IDs_temp)   

        
        
        #return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})
        return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  
    
    def _normalize(self, data):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
                                
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
        return data
   
    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end      


    def __data_generation(self, list_IDs_temp):
        'read the waveforms'         
        X = np.zeros((self.num_part*self.batch_size, self.dim, self.n_channels))   # （27*bacth_size，20000，3）
        y1 = np.zeros((self.num_part*self.batch_size, self.dim, 1))    # labels for detection（200，20000，1）
        y2 = np.zeros((self.num_part*self.batch_size, self.dim, 1))    # labels for P
        y3 = np.zeros((self.num_part*self.batch_size, self.dim, 1))    # labels for S          # diting_0.hdf5 
        
        # DiTing_csv_path = '/nas-alinlp/xiaozhou.zyx/DiTing_datas/DiTing330km_part_0.csv'
        # 试试能不能读取dataset,试过可行
        # dataset = fl.get('earthquake/'+str('000001.0004'))
        # print(list_IDs_temp)
        
    #     dataset = f.get('earthquake/'+str(key))    
    #     data = np.array(dataset).astype(np.float32)

        for part in range(0, len(list_IDs_temp)):                  # len(list_IDs_temp) = 27   
            file_name = self.input_dir + 'DiTing330km_part_{}.hdf5'.format()
            csv_name = self.input_dir + 'DiTing330km_part_{}.csv'.format()
            csv_file = pd.read_csv(csv_name, dtype={'key':str})
            IDs = list_IDs_temp[part]     # 每个part对应取的batch个IDs  ['030154.0629', '017276.0137']
            with h5py.File(file_name, 'r') as f:
                for i, ID in enumerate(IDs):
                    additions = None
                    # with h5py.File(self.file_name, 'r') as f:
                    dataset = f.get('earthquake/'+str(ID))    
                    data = np.array(dataset).astype(np.float32)
                    # #print(type(dataset))  # <class 'h5py._hl.dataset.Dataset'>
      
                    key = csv_file.key
                    meta_dataset = csv_file.loc[key==ID]

                
                    spt = int((meta_dataset['p_pick']+30)*100); # 100HZ  # p_pick仍然是时刻  # p_pick的到时乘上频率，'coda_end_sample']);
                    snr_s_Z_amp = meta_dataset['Z_S_amplitude_snr']
                    snr_s_N_amp = meta_dataset['N_S_amplitude_snr']
                    snr_s_E_amp = meta_dataset['E_S_amplitude_snr'] 
                    snr = np.array([snr_s_E_amp, snr_s_N_amp, snr_s_Z_amp])   # 使用S波，波形振幅的信噪比
                    sst = int((meta_dataset['s_pick']+30)*100);
        

                    coda_end = int(sst+5000);
                
                    
                    
  
                    if self.shift_event_r :
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                     
                    
                    data = self._normalize(data)                          
                    
                
                    X[self.batch_size*part +i, :, :] = data       # data (20000,3)                                  

                    
                    ## labeling 

                    sd = None                             
                    if sst and spt:
                        sd = sst - spt      

                    if sd and sst+int(self.coda_ratio*sd) <= self.dim: 
                        y1[self.batch_size*part +i, spt:int(sst+(self.coda_ratio*sd)), 0] = 1        
                    else:
                        y1[self.batch_size*part +i, spt:self.dim, 0] = 1         
                    if spt: 
                        y2[self.batch_size*part +i, spt, 0] = 1
                    if sst:
                        y3[self.batch_size*part +i, sst, 0] = 1                       
                    
                    if additions:
                        'additions=none'
                        add_sd = None
                        add_spt = additions[0];
                        add_sst = additions[1];
                        if add_spt and add_sst:
                            add_sd = add_sst - add_spt  
                            
                        if add_sd and add_sst+int(self.coda_ratio*add_sd) <= self.dim: 
                            y1[self.batch_size*part +i, add_spt:int(add_sst+(self.coda_ratio*add_sd)), 0] = 1        
                        else:
                            y1[self.batch_size*part +i, add_spt:self.dim, 0] = 1                     
                        if add_spt:
                            y2[self.batch_size*part +i, add_spt, 0] = 1
                        if add_sst:
                            y3[self.batch_size*part +i, add_sst, 0] = 1 
            
        
        #fl.close() 
                           
        return X, y1.astype('float32'), y2.astype('float32'), y3.astype('float32')



class DataGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=50
        Batch size.
            
    n_channels: int, default=3
        Number of channels.

            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input 
    {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=50, 
                 n_channels=3):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()

    # list_IDs是一串波形的ID
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data):  
        'Normalize waveforms in each batch'
        data = data-np.mean(data, axis=0, keepdims=True)
                               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
        
        return data    


    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 

        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            dataset = fl.get('earthquake/'+str(ID))
            data = np.array(dataset)
            
                   
            data = self.normalize(data)  
                            
            #print(data.shape)   # (20000,3)
            #print(X.shape)      # (200, 20000, 3)
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X








def picker(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, spt=None, sst=None):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array: pred_DD_mean[ts]
        Detection peaks probability.

    yh2 : 1D array :pred_PP_mean[ts]
        P arrival probability..  [0 0 ... 1 1 1 ... 0 0 0]
        
    yh3 : 1D array
        S arrival probability.. [0 0 ... 1 1 1 ... 0 0 0] 


    yh1_std : 1D array [1,20000]
        Detection standard deviations. 
        
    yh2_std : 1D array
        P arrival standard deviations.  
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
    spt : {int, None}, default=None    
        P arrival time in sample.
        
    sst : {int, None}, default=None
        S arrival time in sample. 
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
    


    
    pred_DD_ind = K.argmax(yh1, axis=-1)# 概率最大的位置   [2961]   
    pred_PP_ind = K.argmax(yh2, axis=-1)   # P点概率最大的位置
    pred_SS_ind = K.argmax(yh3, axis=-1)

    
    detection = []
    try:
        dd_arr = np.where(yh1>=args['detection_threshold'])[0]   # 概率大于threshold=0.01的index
        detection.append([int(pred_DD_ind),dd_arr[-2]])    # [on, off] indexs       确定一下是pred_DD_ind开始还是从dd_arr[1]
    except:
        detection.append([int(pred_DD_ind),args['input_dimention'][0]])

    
    pp_arr = np.array([pred_PP_ind])    # 一个位置点
    print('pp_arr:{}'.format(pp_arr)) 
    ss_arr = np.array([pred_SS_ind])
    print('ss_arr:{}'.format(ss_arr)) 



    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    
  
    if len(pp_arr) > 0:
        P_uncertainty = None
            
        for pick in range(len(pp_arr)):    # 
            pauto = pp_arr[pick]   # pauto是概率最大的index

                        
            if args['estimate_uncertainty'] :#and pauto:    
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)
                    
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3)   # 取整=1.000
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})          # P_PICKS= {2306: [1.0, 0.0]}      
    else:
        print('len(pp_arr)==0!!')    

    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]    # sauto等于取1的位置下标      
            if args['estimate_uncertainty']:  #and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})          # S_PICKS= {2306: [1.0, 0.0]}
            



    
        
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):     # len(detection) ==  1                          
            if args['estimate_uncertainty']:               
                try:
                    D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][-1]])
                    D_uncertainty = np.round(D_uncertainty, 3)
                except:
                    D_uncertainty = np.round(1.0, 3)
                try:
                    D_prob = np.mean(yh1[detection[ev][0]:detection[ev][-1]])
                    D_prob = np.round(D_prob, 3)
                except:
                    D_prob = np.round(0.0, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][-1]]})            
    
    # matching the detection and picks
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair
    
    #ipdb.set_trace()
    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        
        S_error = None
        P_error = None        
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                #if Ss > bg and Ss < ed:
                candidate_Ss.update({Ss : S_val})    # Ss:sauto   S_val:[S_prob, S_uncertainty]
             
            if len(candidate_Ss) > 1:                
            
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    #if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                    candidate_Ps.update({Ps : P_val}) 
                else:         
                    #if Ps > bg-100 and Ps < ed:
                    candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}
                    

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:   

                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                
                if sst and sst > bg and sst < EVENTS[ev][2]:
                    if list(candidate_Ss)[0]:
                        S_error = sst -list(candidate_Ss)[0] 

                    else:
                        S_error = None
                                            

                if list(candidate_Ps)[0]:  
                    P_error = spt - list(candidate_Ps)[0] 
                        
                else:
                    P_error = None
                                          
                pick_errors.update({bg:[P_error, S_error]})
                
    
    # matches.update({bg:[ed, EVENTS[ev][0], EVENTS[ev][1],list(P_PICKS)[0], P_PICKS[list(P_PICKS)[0]][0],  P_PICKS[list(P_PICKS)[0]][1], list(S_PICKS)[0], S_PICKS[list(S_PICKS)[0]][0], S_PICKS[list(S_PICKS)[0]][1]]})
    # # print('matches:{}'.format(matches))
    return matches, pick_errors, yh3

def picker_prediction(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array: pred_DD_mean[ts]
        Detection peaks probability.

    yh2 : 1D array :pred_PP_mean[ts]
        P arrival probability..  [0 0 ... 1 1 1 ... 0 0 0]
        
    yh3 : 1D array
        S arrival probability.. [0 0 ... 1 1 1 ... 0 0 0] 


    yh1_std : 1D array [1,20000]
        Detection standard deviations. 
        
    yh2_std : 1D array
        P arrival standard deviations.  
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
    

    
    pred_DD_ind = K.argmax(yh1, axis=-1)# 概率最大的位置   [2961]   
    pred_PP_ind = K.argmax(yh2, axis=-1)   # P点概率最大的位置
    pred_SS_ind = K.argmax(yh3, axis=-1)

    
    detection = []
    try:
        dd_arr = np.where(yh1>=args['detection_threshold'])[0]   # 概率大于threshold=0.01的index
        detection.append([int(pred_DD_ind),dd_arr[-2]])    # [on, off] indexs       确定一下是pred_DD_ind开始还是从dd_arr[1]
    except:
        detection.append([int(pred_DD_ind),args['input_dimention'][0]])
    
    pp_arr = np.array([pred_PP_ind])    # 一个位置点
    #print('pp_arr:{}'.format(pp_arr)) 
    ss_arr = np.array([pred_SS_ind])
    #print('ss_arr:{}'.format(ss_arr)) 
    

    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    
  
    if len(pp_arr) > 0:
        P_uncertainty = None
            
        for pick in range(len(pp_arr)):    # 
            pauto = pp_arr[pick]   # pauto是概率最大的index

                        
            if args['estimate_uncertainty'] :#and pauto:    
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)
                    
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3)   # 取整=1.000
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})          # P_PICKS= {2306: [1.0, 0.0]}      
    else:
        print('len(pp_arr)==0!!')    

    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]    # sauto等于取1的位置下标      
            if args['estimate_uncertainty']:  #and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})          # S_PICKS= {2306: [1.0, 0.0]}
            



    
        
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):     # len(detection) ==  1                          
            if args['estimate_uncertainty']:               
                try:
                    D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][-1]])
                    D_uncertainty = np.round(D_uncertainty, 3)
                except:
                    D_uncertainty = np.round(1.0, 3)
                try:
                    D_prob = np.mean(yh1[detection[ev][0]:detection[ev][-1]])
                    D_prob = np.round(D_prob, 3)
                except:
                    D_prob = np.round(0.0, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][-1]]})            
    
   

    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
               
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                #if Ss > bg and Ss < ed:
                candidate_Ss.update({Ss : S_val})    # Ss:sauto   S_val:[S_prob, S_uncertainty]
             
            if len(candidate_Ss) > 1:                
             
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    #if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                    candidate_Ps.update({Ps : P_val}) 
                else:         
                    #if Ps > bg-100 and Ps < ed:
                    candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}
                    
                    

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:   

                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                
    return matches



def generate_arrays_from_file(file_list, step):
    
    """ 
    
    Make a generator to generate list of trace names.
    
    Parameters
    ----------
    file_list : str
        A list of trace names.  
        
    step : int
        Batch size.  
        
    Returns
    --------  
    chunck : str
        A batch of trace names. 
        
    """     
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   

    
    

def acc(y_true, y_pred):
    
    """ 
    
    Calculate acc.
    
    Parameters
    ----------
    y_true : 1D array
        ce labeling: 
        detector[0,0,..,1,1,1,1,...0,0]
        p[0,0,...,0,1,0,...,0]
        s[0,0,...,0,0,1,...,0]


        
    y_pred : 1D array
        Predicted labels.   [pro,pro,...,pro]  
        
    Returns
    -------  
    acc : float
        Calculated accuracy. 
        
    """     
    '''
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    '''

    def precision(y_true, y_pred):
        'Precison metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    """
    def accuracy(y_true, y_pred):
        'compute accuracy'
        #ipdb.set_trace()
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        
        y_true_ind = np.where(y_true==1)[0]   #  ture arrival
        
        if len(y_true_ind) == 1:
            'pick accuracy'
            y_pred_ind = K.argmax(y_pred, axis=-1)   # predicted arrival
            err = abs(y_pred_ind - y_true_ind)      # distance
            err = np.where(err<=100, 1,0)
            accuracy = sum(err)/len(err)
        else:
            'detector accuracy'
            accuracy = None
            '''
            y_true_ind = np.array(y_true_ind[0],y_true_ind[-1])    # 真实的开始和结束的位置
            dd_arr = np.where(y_pred>=args['detection_threshold'])[0]   # 概率大于threshold=0.01的index
            # [ 0,  2756,  2757,  2758,  2759,  2760,  2761,  2762,  2763,...,  3319, 19999]
            y_pred_ind = np.array([dd_arr[1],dd_arr[-2]]) 
            err = abs(y_pred_ind - y_true_ind)
            '''
            
        return accuracy
    """


    #acc = accuracy(y_true, y_pred)
    #recall = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return prec


def normalize(data):
    
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """   
    
    data -= np.mean(data, axis=0, keepdims=True)             
       
    std_data = np.std(data, axis=0, keepdims=True)
    assert(std_data.shape[-1] == data.shape[-1])
    std_data[std_data == 0] = 1
    data /= std_data
    return data
    
    

  
class LayerNormalization(keras.layers.Layer):
    
    """ 
    
    Layer normalization layer modified from https://github.com/CyberZHG based on [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
    
    Parameters
    ----------
    center: bool
        Add an offset parameter if it is True. 
        
    scale: bool
        Add a scale parameter if it is True.     
        
    epsilon: bool
        Epsilon for calculating variance.     
        
    gamma_initializer: str
        Initializer for the gamma weight.     
        
    beta_initializer: str
        Initializer for the beta weight.     
                    
    Returns
    -------  
    data: 3D tensor
        with shape: (batch_size, …, input_dim) 
            
    """   
              
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):

        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
      

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    
    
class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer. modified from https://github.com/CyberZHG 
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """
    
    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y


class SeqSelfAttention(keras.layers.Layer):
    """Layer initialization. modified from https://github.com/CyberZHG
    For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
    :param units: The dimension of the vectors that used to calculate the attention weights.
    :param attention_width: The width of local attention.
    :param attention_type: 'additive' or 'multiplicative'.
    :param return_attention: Whether to return the attention weights for visualization.
    :param history_only: Only use historical pieces of data.
    :param kernel_initializer: The initializer for weight matrices.
    :param bias_initializer: The initializer for biases.
    :param kernel_regularizer: The regularization for weight matrices.
    :param bias_regularizer: The regularization for biases.
    :param kernel_constraint: The constraint for weight matrices.
    :param bias_constraint: The constraint for biases.
    :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                              in additive mode.
    :param use_attention_bias: Whether to use bias while calculating the weights of attention.
    :param attention_activation: The activation used for calculating the weights of attention.
    :param attention_regularizer_weight: The weights of attention regularizer.
    :param kwargs: Parameters for parent class.
    """
        
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        input_len = inputs.get_shape().as_list()[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}



def _block_BiLSTM(filters, drop_rate, padding, inpR):
    'Returns LSTM residual block'    
    prev = inpR
    x_rnn = Bidirectional(LSTM(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate))(prev)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out


def _block_CNN_1(filters, ker, drop_rate, activation, padding, inpC): 
    ' Returns CNN residual blocks '
    prev = inpC
    layer_1 = BatchNormalization()(prev) 
    act_1 = Activation(activation)(layer_1) 
    act_1 = SpatialDropout1D(drop_rate)(act_1, training=True)
    conv_1 = Conv1D(filters, ker, padding = padding)(act_1) 
    
    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation(activation)(layer_2) 
    act_2 = SpatialDropout1D(drop_rate)(act_2, training=True)
    conv_2 = Conv1D(filters, ker, padding = padding)(act_2)
    
    res_out = add([prev, conv_2])
    
    return res_out 


def _transformer(drop_rate, width, name, inpC): 
    ' Returns a transformer block containing one addetive attention and one feed  forward layer with residual connections '
    x = inpC
    
    att_layer, weight = SeqSelfAttention(return_attention =True,                                       
                                         attention_width = width,
                                         name=name)(x)
   
#  att_layer = Dropout(drop_rate)(att_layer, training=True)    
    att_layer2 = add([x, att_layer])    
    norm_layer = LayerNormalization()(att_layer2)
    
    FF = FeedForward(units=128, dropout_rate=drop_rate)(norm_layer)
    
    FF_add = add([norm_layer, FF])    
    norm_out = LayerNormalization()(FF_add)
    
    return norm_out, weight 

     

def _encoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the encoder that is a combination of residual blocks and maxpooling.'        
    e = inpC
    for dp in range(depth):
        e = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(e)             
        e = MaxPooling1D(2, padding = padding)(e)            
    return(e) 


def _decoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the dencoder that is a combination of residual blocks and upsampling. '           
    d = inpC
    for dp in range(depth):        
        d = UpSampling1D(2)(d) 
        if dp == 3:
            d = Cropping1D(cropping=(6, 6))(d)       # d = Cropping1D(cropping=(1, 1))(d)     
        d = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(d)        
    return(d)  
 


def _lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



class cred2():
    
    """ 
    
    Creates the model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.

    activationf: str
        Activation funciton type.

    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.

    cnn_blocks: int
        The number of residual CNN blocks.

    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.

    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 

    kernel_regularizer: str
        l1 norm regularizer.

    bias_regularizer: str
        l1 norm regularizer.
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['cross_entropy', 'cross_entropy', 'cross_entropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 

        
    def __call__(self, inp):

        x = inp
    
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)

            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                             
                            encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)
        # print('d:{}'.format(d))
        # print('d.shape:{}'.format(d.shape))

        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                attention_width= 3,
                                                name='attentionP')(PLSTM)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        # print('P:{}'.format(P))
        # print('P.shape:{}'.format(P.shape))
        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded) 
        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                attention_width= 3,
                                                name='attentionS')(SLSTM)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        # print('S:{}'.format(S))
        # print('S.shape:{}'.format(S.shape))

        
        model = Model(inputs=inp, outputs=[d, P, S])
        # model.compile(loss=self.loss_types, loss_weights=self.loss_weights, optimizer=Adam(lr=_lr_schedule(0)), metrics=['acc'])

        return model
