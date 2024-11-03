#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modified on Fri Oct 28 14:32:14 2022

@author: xiaozhou.zyx
last update: 

# 457line: trace_name -> key
"""

from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
import ipdb

from tensorflow.keras.optimizers import Adam

import shutil
import multiprocessing
from .Method import DataGenerator, _lr_schedule, cred2, acc
import datetime
from tqdm import tqdm
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def trainer(input_dir=None,
            num_part=None,
            output_name=None,                
            input_dimention=(20000, 3),
            cnn_blocks=5,
            lstm_blocks=2,
            padding='same',
            activation = 'relu',            
            drop_rate=0.1,
            shuffle=True, 
            shift_event_r=0.99,
            coda_ratio=0.4,                
            loss_weights=[0.05, 0.40, 0.55],
            loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],     # 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'
            train_valid_test_split=[0.85, 0.05, 0.10],
            batch_size=200,
            epochs=200, 
            monitor='val_loss',
            patience=12,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=False):
        
    """
    
    Generate a model and train it.  
    
    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of data with NumPy arrays containing 3 component waveforms each 1 min long.

    input_csv: str, default=None
        Path to a CSV file with one column (trace_name) listing the name of all datasets in the hdf5 file.

    num_part: int, default=None
        number of parts of files
    
    output_name: str, default=None
        Training Output directory, 'TrainResults'.
        
    input_dimention: tuple, default=(6000, 3)
        Loss types for detection, P picking, and S picking respectively. 
        
    cnn_blocks: int, default=5
        The number of residual blocks of convolutional layers.
        
    lstm_blocks: int, default=2
        The number of residual blocks of BiLSTM layers.
        
    padding: str, default='same'
        Padding type.
        
    activation: str, default='relu'
        Activation function used in the hidden layers.

    drop_rate: float, default=0.1
        Dropout value.

    shuffle: bool, default=True
        To shuffle the list prior to the training.

    normalization_mode: str, default='std'
        Mode of normalization for data preprocessing, 'max': maximum amplitude among three components, 'std', standard deviation. 

    augmentation: bool, default=True
        If True, data will be augmented simultaneously during the training.


    shift_event_r: float, default=0.99
        Rate of augmentation for randomly shifting the event within a trace.
     

    coda_ratio: float, defaults=0.4
        % of S-P time to extend event/coda envelope past S pick.
        
        
    loss_weights: list, defaults=[0.03, 0.40, 0.58]
        Loss weights for detection, P picking, and S picking respectively.
              
    loss_types: list, defaults=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'] 
        Loss types for detection, P picking, and S picking respectively.  
        
    train_valid_test_split: list, defaults=[0.85, 0.05, 0.10]
        Precentage of data split into the training, validation, and test sets respectively. 
         
    batch_size: int, default=200
        Batch size of every part
          
    epochs: int, default=200
        The number of epochs.
          
    monitor: int, default='val_loss'
        The measure used for monitoring.
           
    patience: int, default=12
        The number of epochs without any improvement in the monitoring measure to automatically stop the training.          
           
    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None. 
         
    gpu_limit: float, default=None
        Set the maximum percentage of memory usage for the GPU.
        
    use_multiprocessing: bool, default=True
        If True, multiple CPUs will be used for the preprocessing of data even when GPU is used for the prediction. 

    Returns
    -------- 
    output_name/models/output_name_.h5: This is where all good models will be saved.  
    
    output_name/final_model.h5: This is the full model for the last epoch.
    
    output_name/model_weights.h5: These are the weights for the last model.

    output_name/X_report.txt: A summary of the parameters used for prediction and performance.
    
    output_name/test.npy: A number list containing the trace names for the test set.
    
    output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.
    
    output_name/X_learning_curve_loss.png: The learning curve of loss.

        
    """     


    args = {
    "input_dir": input_dir,
    "num_part":num_part,
    "output_name": output_name,
    "input_dimention": input_dimention,
    "cnn_blocks": cnn_blocks,
    "lstm_blocks": lstm_blocks,
    "padding": padding,
    "activation": activation,
    "drop_rate": drop_rate,
    "shuffle": shuffle,
    "shift_event_r": shift_event_r,
    "coda_ratio": coda_ratio,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "train_valid_test_split": train_valid_test_split,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience,                    
    "gpuid": gpuid,
    "gpu_limit": gpu_limit,
    "use_multiprocessing": use_multiprocessing
    }
                       
    def train(args):
        """ 
        
        Performs the training.
    
        Parameters
        ----------
        args : dic
            A dictionary object containing all of the input parameters. 

        Returns
        -------
        history: dic
            Training history.  
            
        model: 
            Trained model.
            
        start_training: datetime
            Training start time. 
            
        end_training: datetime
            Training end time. 
            
        save_dir: str
            Path to the output directory. 
            
        save_models: str
            Path to the folder for saveing the models.  
            
        training size: int
            Number of training samples.
            
        validation size: int
            Number of validation samples.  
            
        """    

        
        save_dir, save_models=_make_dir(args['output_name'])
         
        #ipdb.set_trace()
        callbacks=_make_callback(args, save_models)
        # gpus = tf.config.list_logical_devices('GPU')
        # strategy = tf.distribute.MirroredStrategy(gpus)
        # with strategy.scope():
        model=_build_model(args)
        model.compile(loss=args['loss_types'], loss_weights=args['loss_weights'], optimizer=Adam(lr=_lr_schedule(0)), metrics=[acc])
        

        
        if args['gpuid']:           
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
            tf.Session(config=tf.ConfigProto(log_device_placement=True))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
            K.tensorflow_backend.set_session(tf.Session(config=config))
        
        start_training = time.time()   
    
            
            
                
        training = []
        validation = []

        
        #ipdb.set_trace()    
        for part in range(args['num_part']):
            # file_name, csv_name = get_file_name(args, part) 
            training_, validation_=_split(args, part,  save_dir=save_dir)       # part list of IDs of training,testing validation
            # len(training) = 60000 
            training.extend(training_)       # all IDs from 27 files ， len = 60000*27
            validation.extend(validation_)

        params_training = {'input_dir': args['input_dir'],
                            'num_part': args['num_part'],
                        'dim': args['input_dimention'][0],
                        'batch_size': args['batch_size'],
                        'n_channels': args['input_dimention'][-1],
                        'shuffle': args['shuffle'],  
                        'norm_mode': args['normalization_mode'],
                        'augmentation': args['augmentation'],
                        'coda_ratio': args['coda_ratio'],
                        'shift_event_r': args['shift_event_r']}
                    
        params_validation = {'input_dir': args['input_dir'],
                            'num_part': args['num_part'],
                            'dim': args['input_dimention'][0],
                            'batch_size': args['batch_size'],
                            'n_channels': args['input_dimention'][-1],
                            'shuffle': False,  
                            'norm_mode': args['normalization_mode'],
                            'augmentation': False}
        


        training_generator = DataGenerator(training, **params_training)      # <EQTransformer.core.EqT_utils.DataGenerator object at 0x7f89183a9550>
        validation_generator = DataGenerator(validation,  **params_validation)

        

        print('Started training  ...') 
        history = model.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    use_multiprocessing=args['use_multiprocessing'],
                                    workers=16,    
                                    callbacks=callbacks, 
                                    epochs=args['epochs'],
                                    class_weight=None)
        
                

        end_training = time.time()  
        
        return history, model, start_training, end_training, save_dir, save_models, len(training), len(validation)
 
    history, model, start_training, end_training, save_dir, save_models, training_size, validation_size=train(args)  # 得到训练结果
    _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args)
    # 把训练结果可视化出来且写到文件里，return“output_name/··“
    # history.history.keys()
    # dict_keys(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss',
    # 'detector_accuracy', 'picker_P_accuracy', 'picker_S_accuracy', 
    # 'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss', 
    # 'val_detector_accuracy', 'val_picker_P_accuracy', 'val_picker_S_accuracy', 'lr'])

                
# def _batch_all_(args, part, list_IDs, X, y1, y2, y3, **kwargs):
#     """
#     每个part的batch放到一起组成一整个batch
#     """
#     ipdb.set_trace() 
#     X_ = DataGenerator(list_IDs, **kwargs)
    
        
#     X[(part-1)*args['batch_size']:part*args['batch_size'],:,:] = X_ 
#     y1[(part-1)*args['batch_size']:part*args['batch_size'],:,:] = y1_
#     y2[(part-1)*args['batch_size']:part*args['batch_size'],:,:] = y2_
#     y3[(part-1)*args['batch_size']:part*args['batch_size'],:,:] = y3_       #所有data的batch放到一起组成一整个batch

    
#     return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})


'''
def get_file_name(args,
                    part,
                    ):

    """
    Input:
    part: batch of IDs part 

    Output:
    data
    """
    # with h5py.File(args['input_dir'] + 'DiTing330km_part_{}.hdf5'.format(part), 'r') as f:
    #     dataset = f.get('earthquake/'+str(key))    
    #     data = np.array(dataset).astype(np.float32)
    file_name = args['input_dir'] + 'DiTing330km_part_{}.hdf5'.format(part)
    csv_name = args['input_dir'] + 'DiTing330km_part_{}.csv'.format(part)

    return file_name, csv_name
'''

def _make_dir(output_name): 
    """ 
    
    Make the output directories.

    Parameters
    ----------
    output_name: str
        Name of the output directory.
                   
    Returns
    -------   
    save_dir: str
        Full path to the output directory.
        
    save_models: str
        Full path to the model directory. 
        
    """   
    
    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name))
        save_models = os.path.join(save_dir, 'models')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
    return save_dir, save_models



def _build_model(args): 
    
    """ 
    
    Build and compile the model.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
               
    Returns
    -------   
    model: 
        Compiled model.
        
    """       
    
    inp = Input(shape=args['input_dimention'], name='input') 
    model = cred2(nb_filters=[8, 16, 16, 32, 32, 64, 64],
              kernel_size=[11, 9, 7, 7, 5, 5, 3],
              padding=args['padding'],
              activationf =args['activation'],
              cnn_blocks=args['cnn_blocks'],
              BiLSTM_blocks=args['lstm_blocks'],
              drop_rate=args['drop_rate'], 
              loss_weights=args['loss_weights'],
              loss_types=args['loss_types'],
              kernel_regularizer=keras.regularizers.l2(1e-6),
              bias_regularizer=keras.regularizers.l1(1e-4)
               )(inp)  
    model.summary()  
    return model  
    


def _split(args, part, save_dir):
    
    """ 
    
    Split the list of input data into training, validation, and test set.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_dir: str
       Path to the output directory. 
       '/nas-alinlp/xiaozhou.zyx/eqt_on_DiTing/EQT_on_DiTing/test_trainer_outputs/'
              
    Returns
    -------   
    training: str
        List of keys(IDs) for the training set. 
    validation : str
        List of keys(IDs) for the validation set. 
                
    """  
    
    df = pd.read_csv(args['input_dir'] + 'DiTing330km_part_{}.csv'.format(1), dtype={'key':str}) 
    key = df['key']
    key_correct = [x.split('.') for x in key]
    #print(key_correct)
    for i in key_correct:
        #print(i)
        key_correct = i[0].rjust(6,'0')+ '.' + i[1].ljust(4,'0')
        key_correct = ['.'.join(i)]
    ev_list = key.tolist()    # ev_list是每条波形的唯一索引，对应谛听里的‘key’
    np.random.shuffle(ev_list)     
    training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]         # len = 60000
    validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
                            int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]    # len = 20000
    test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    np.save(save_dir+'/test'+str(part), test)  
    return training, validation



def _make_callback(args, save_models):
    
    """ 
    
    Generate the callback.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_models: str
       Path to the output directory for the models. 
              
    Returns
    -------   
    callbacks: obj
        List of callback objects. 
        
        
    """    
    
    m_name=str(args['output_name'])+'_{epoch:03d}.h5'   
    filepath=os.path.join(save_models, m_name)  
    early_stopping_monitor=EarlyStopping(monitor=args['monitor'], 
                                           patience=args['patience']) 
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=args['monitor'], 
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=False)  
    lr_scheduler=LearningRateScheduler(_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args['patience']-2,
                                   min_lr=0.5e-6)

    #callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    return callbacks
 
    





def _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args): 

    """ 
    
    Write down the training results.

    Parameters
    ----------
    history: dic
        Training history.  
   
    model: 
        Trained model.  

    start_training: datetime
        Training start time. 

    end_training: datetime
        Training end time.    
         
    save_dir: str
        Path to the output directory. 

    save_models: str
        Path to the folder for saveing the models.  
      
    training_size: int
        Number of training samples.    

    validation_size: int
        Number of validation samples. 

    args: dic
        A dictionary containing all of the input parameters. 
              
    Returns
    -------- 
    

    ./output_name/X_report.txt: A summary of parameters used for the prediction and perfomance.

    ./output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.         

    ./output_name/X_learning_curve_loss.png: The learning curve of loss.  
        
        
    """   
    

    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/model_weights.h5')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['detector_loss'])
    ax.plot(history.history['picker_P_loss'])
    ax.plot(history.history['picker_S_loss'])
    try:
        ax.plot(history.history['val_loss'], '--')
        ax.plot(history.history['val_detector_loss'], '--')
        ax.plot(history.history['val_picker_P_loss'], '--')
        ax.plot(history.history['val_picker_S_loss'], '--') 
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss', 
               'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss'], loc='upper right')
    except Exception:
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss'], loc='upper right')  
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
       
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['detector_acc'])
    ax.plot(history.history['picker_P_acc'])
    ax.plot(history.history['picker_S_acc'])
    try:
        ax.plot(history.history['val_detector_acc'], '--')
        ax.plot(history.history['val_picker_P_acc'], '--')
        ax.plot(history.history['val_picker_S_acc'], '--')
        ax.legend(['detector_acc', 'picker_P_acc', 'picker_S_acc', 'val_detector_acc', 'val_picker_P_acc', 'val_picker_S_acc'], loc='lower right')
    except Exception:
        ax.legend(['detector_acc', 'picker_P_acc', 'picker_S_acc'], loc='lower right')        
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_acc.png'))) 

    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta    
    
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')         
        the_file.write('input_dir: '+str(args['input_dir'])+'\n')            
        the_file.write('num_part: '+str(args['num_part'])+'\n')
        the_file.write('output_name: '+str(args['output_name'])+'\n')  
        the_file.write('================== Model Parameters ========================='+'\n')   
        the_file.write('input_dimention: '+str(args['input_dimention'])+'\n')
        the_file.write('cnn_blocks: '+str(args['cnn_blocks'])+'\n')
        the_file.write('lstm_blocks: '+str(args['lstm_blocks'])+'\n')
        the_file.write('padding_type: '+str(args['padding'])+'\n')
        the_file.write('activation_type: '+str(args['activation'])+'\n')        
        the_file.write('drop_rate: '+str(args['drop_rate'])+'\n')            
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')  
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('epochs: '+str(args['epochs'])+'\n')   
        the_file.write('train_valid_test_split: '+str(args['train_valid_test_split'])+'\n')           
        the_file.write('total number of training: '+str(training_size)+'\n')
        the_file.write('total number of validation: '+str(validation_size)+'\n')
        the_file.write('monitor: '+str(args['monitor'])+'\n')
        the_file.write('patience: '+str(args['patience'])+'\n') 
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')  
        the_file.write('================== Training Performance ====================='+'\n')  
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))                         
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last detector_loss: '+str(history.history['detector_loss'][-1])+'\n')
        the_file.write('last picker_P_loss: '+str(history.history['picker_P_loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['picker_S_loss'][-1])+'\n')
        the_file.write('last detector_acc: '+str(history.history['detector_acc'][-1])+'\n')
        the_file.write('last picker_P_acc: '+str(history.history['picker_P_acc'][-1])+'\n')
        the_file.write('last picker_S_acc: '+str(history.history['picker_S_acc'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('augmentation: '+str(args['augmentation'])+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')               
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('shift_event_r: '+str(args['shift_event_r'])+'\n')                            
        the_file.write('coda_ratio: '+str(args['coda_ratio'])+'\n')          

