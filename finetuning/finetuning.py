from keras.callbacks import ModelCheckpoint
from data_generator import train_datagenerator
import scipy.io as scio 
from scipy import signal
from keras.models import Model,load_model
from keras.layers import Input
import numpy as np
from random import sample
import os
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
    # read the data
    data = scio.loadmat(path)
    # get the EEG-data of the selected electrodes and downsampling it
    data_1 = data['data']
    c1 = [47,53,54,55,56,57,60,61,62]
    
    train_data = data_1[c1,:,:,:]
    # get the filtered EEG-data with six-order Butterworth filter of the first sub-filter
    block_data_list1 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn11,wn21], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list1.append(target_data_list)
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
    block_data_list2 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn12,wn22], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list2.append(target_data_list)

    block_data_list3 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn13,wn23], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list3.append(target_data_list) 
    return block_data_list1, block_data_list2, block_data_list3

if __name__ == '__main__':
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    #%% Setting hyper-parameters
    # downsampling coefficient and sampling frequency after downsampling
    fs = 250
    # the number of the electrode channels
    channel = 9
    # the hyper-parameters of the training process
    batchsize = 256
    
    # the filter ranges of the four sub-filters in the filter bank
    f_down1 = 6
    f_up1 = 50
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 14
    f_up2 = 50
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 22
    f_up3 = 50
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs

    #%% Training the models of multi-subjects and multi-time-window
    # the list of the time-window
    t_train_list = [1.0] # 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2
    # selecting the number of training blocks
    training_block_list = [[[1], [1, 4], [4, 5, 3], [3, 1, 2, 5], [3, 2, 4, 1, 5]], [[0], [3, 4], [5, 3, 0], [0, 5, 4, 2], [5, 0, 4, 3, 2]], [[5], [4, 1], [4, 5, 1], [1, 5, 3, 4], [5, 4, 1, 3, 0]], [[0], [4, 2], [1, 0, 5], [4, 2, 5, 1], [1, 2, 4, 5, 0]], [[3], [2, 1], [5, 0, 2], [1, 0, 5, 3], [2, 0, 1, 5, 3]], [[2], [4, 3], [4, 3, 0], [2, 0, 4, 1], [1, 0, 3, 4, 2]]]
    # list of all categiories
    all_target_list = list(range(40))
    # list of randomly selected 32 categiories
    unseen_8_target_list = [13, 6, 23, 37, 35, 20, 2, 39, 24, 27, 33, 10, 16, 0, 28, 32, 1, 14, 3, 29, 4, 22, 18, 12, 30, 19, 17, 25, 9, 21, 31, 8]
    # list of randomly selected 24 categiories
    unseen_16_target_list = [32, 27, 28, 20, 1, 36, 39, 26, 35, 17, 14, 21, 33, 18, 3, 37, 24, 23, 34, 16, 10, 8, 7, 12]
    # list of randomly selected 16 categiories
    unseen_24_target_list = [37, 23, 5, 22, 24, 16, 35, 15, 6, 27, 17, 10, 4, 2, 14, 39]
    # list of randomly selected 8 categiories
    unseen_32_target_list = [30, 8, 35, 20, 6, 5, 37, 26]
    # list of different numbers of unseen (seen) categories
    unseen_list = [all_target_list,unseen_8_target_list,unseen_16_target_list,unseen_24_target_list,unseen_32_target_list]

    for group_n in range(5):
        test_subject_list = list(range(group_n*7+1,(group_n+1)*7+1))
        # if group_n == 1:
        # test_subject_list = [22]
        # test_subject_list = list(range(3,8))
        for sub_selelct in test_subject_list:#sub_list:
        # the path of the dataset and you need change it for your training
            path = '/data/dwl/ssvep/benchmark/S%d.mat'%sub_selelct
            # get the filtered EEG-data of four sub-input, label and the start time of all trials of the training data
            data1, data2, data3 = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
            # different numbers of unseen categories
            for n_unseen in range(5):
                target_list = unseen_list[n_unseen]
                print(target_list)
                
                # selecting the training time-window
                for t_train in t_train_list:
                    # transfer time to frame
                    win_train = int(fs*t_train)
                    # the traing data is randomly divided in the traning dataset and validation set according to the radio of 9:1
                    for test_block in range(5,6):
                        for block_num in range(5,6):
                            train_list = training_block_list[test_block][block_num-1]
                            val_list = [test_block]
                            train_gen = train_datagenerator(batchsize,data1, data2, data3,win_train,train_list, channel,target_list)#, t_train)
                            #%% setting the input of the network
                            teacher_model_path = '/data/dwl/ssvep/model/benchmark_transfer/cnnformer_original_train_loss/tt_%3.1fs_02_%d.h5'%(t_train,group_n)
                            model = load_model(teacher_model_path)# the path of the saved model and you need to change it
                            for layer in model.layers:
                                layer.trainable = False
                            
                            # ############ 4,37,39, and 98 respetively denotes the strategy 1, 2, 3, and 4
                            for i in range(37+1):
                                model.layers[i].trainable = True
                            train_epoch = 10*block_num
                            model_path = '/data/dwl/ssvep/model/benchmark_test/ft_original_unseen_0402_%3.1f/cnnformer_%dblockn_%3.1fs_%d_block%d_unseen_%d.h5'%(t_train,block_num,t_train, sub_selelct,test_block,n_unseen)
                            # some hyper-parameters in the training process
                            model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
                            model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
                            # training
                            history = model.fit_generator(
                                    train_gen,
                                    steps_per_epoch= 10,
                                    epochs=train_epoch,
                                    validation_data=None,
                                    validation_steps=1,
                                    callbacks=[model_checkpoint]
                                    )
    # # show the process of the taining
    # epochs=range(len(history.history['loss']))
    # plt.subplot(221)
    # plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
    # plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
    # plt.title('Traing and Validation accuracy')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_V3.1_acc1.jpg')
    
    # plt.subplot(222)
    # plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    # plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_2.5s_loss0%d.jpg'%sub_selelct)
    
    # plt.show()






# %%
