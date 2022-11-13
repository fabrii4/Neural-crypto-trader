from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dropout, Lambda, concatenate, LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from tcn import TCN, tcn_full_summary
import numpy as np
#from sklearn.model_selection import train_test_split
from termcolor import colored
import csv
import sys
import os
import random
import json
import math

#do not use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings
warnings.filterwarnings("ignore")


def catenate(A, B):
    return B if A.size == 0 else np.concatenate((A, B))

class Scaler:
    #pM=p0*(1+l), pM1=p01*(1+l1)
    def __init__(self, p0, p01, l, l1):
        self.p0 = p0
        self.p01 = p01
        self.scal = p01*l1/(p0*l)
        self.inv_scal=p0*l/(p01*l1)
    
    #scale single point
    def scale(self, p):
        return self.p01+self.scal*(p-self.p0)
    def inv_scale(self, p):
        return self.p0+self.inv_scal*(p-self.p01)

    #scale series
    def transform(self, series):
        return [self.scale(series[i]) for i in range(len(series))]
    def inverse_transform(self, series):
        return [self.inv_scale(series[i]) for i in range(len(series))]


#########################
#      DATASET          #
#########################
#Paths
coin_list=['BTCUSDT', 'LTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'BTCEUR', 'LTCEUR', 'XRPEUR', 'ETHEUR', 'DOGEEUR']
#coin_list=['DOGEEUR', 'LTCEUR']
#coin_list=['LTCUSDT']

X_train=np.array([])
X_train_cat=np.array([])
X_test=np.array([])
X_test_cat=np.array([])
scaler_list_train=np.array([])
scaler_list_test=np.array([])

#type buy/sell
net_type='buy'
if len(sys.argv)>2 and sys.argv[2] == 'sell':
    net_type='sell'

data_folder="../../data/"
timename='5m'
N_buf=100 #number of different strategies per coin
#if len(sys.argv)>1 and sys.argv[1] == 'test':
#    N_buf=1

#parameters
n_fut=16
n_past=128 #used to scale data and buy price
n_past_out=128 #feeded to neural network
n_cat=5
n_state=2
n_actions=2
q_thresh_dict={'buy': 0.2, 'sell': 0}

#Load time series datasets
for i_coin in range(len(coin_list)):
    coin=coin_list[i_coin]
    print(colored('\nLoad Dataset','cyan'))
    #if len(sys.argv)>2: 
    #    timename=sys.argv[2]
    filename='prediction_dataset_'+timename+'_'+coin+'.csv'
    dataset_folder=data_folder+coin+"/action_training/"
    print("Loading file: "+filename)
    #state dataset (future trend prediction + past prices + category prediction)
    dataset = np.genfromtxt(dataset_folder+filename, delimiter=',')
    # flip dataset, time increases along axis 0 (vertical) and decreases along axis 1
    #...t+1, t_0, t-1....
    #...t+2, t+1, t_0,... 
    dataset = np.flip(dataset, axis=0)
    #remove true future and categories
    dataset=dataset[:,n_fut:-n_cat]
    #split dataset time series and categories
    dataset_fut_past=dataset[:,:n_fut+n_past]
    dataset_fut_past = np.reshape(dataset_fut_past,(-1,n_fut+n_past,1))
    dataset_cat=dataset[:,-n_cat:]
    #convert probabilities to categories
    dataset_cat=np.round(dataset_cat)

    #scale temporal data
    scaler_list_coin=[]
    for i in range(len(dataset)):
        #scaler = MinMaxScaler((0.33,0.66))
        #scaler.fit(dataset_fut_past[i,n_fut:])
        scaler=Scaler(dataset_fut_past[i][n_fut][0],0.5,0.05,0.5)
        dataset_fut_past[i]=scaler.transform(dataset_fut_past[i])
        scaler_list_coin.append(scaler)

    #reduce past history length
    dataset_fut_past=dataset_fut_past[:,:n_fut+n_past_out]

    #generate train and test dataset
    N_test=int(len(dataset)*0.1)
    X_train_coin=dataset_fut_past[:-N_test]
    X_test_coin=dataset_fut_past[-N_test:]
    X_train_cat_coin=dataset_cat[:-N_test]
    X_test_cat_coin=dataset_cat[-N_test:]
    scaler_list_coin=np.array(scaler_list_coin)
    scaler_list_train_coin=scaler_list_coin[:-N_test]
    scaler_list_test_coin=scaler_list_coin[-N_test:]
    
    #add coin dataset to dataset coin list
    X_train = catenate(X_train, X_train_coin)
    X_test = catenate(X_test, X_test_coin)
    X_train_cat = catenate(X_train_cat, X_train_cat_coin)
    X_test_cat = catenate(X_test_cat, X_test_cat_coin)
    scaler_list_train = catenate(scaler_list_train, scaler_list_train_coin)
    scaler_list_test = catenate(scaler_list_test, scaler_list_test_coin)


########################################
#Load learning buffers
print(colored(f'\nLoad {net_type} Learning Buffers','cyan'))
X_train_buf=np.array([])
X_test_buf=np.array([])
#Y_train_q=np.array([])
#Y_test_q=np.array([])
Y_train_cat=np.array([])
Y_test_cat=np.array([])
q_threshold=q_thresh_dict[net_type]
for i_coin in range(len(coin_list)):
    coin=coin_list[i_coin]
    dataset_folder=data_folder+coin+"/action_training/"
    filename=net_type+'_'+coin+'_'+timename+'.csv'
    print("Loading file: "+filename)
    #state dataset (future trend prediction + past prices + category prediction)
    dataset = np.genfromtxt(dataset_folder+filename, delimiter=',')
    states=np.full((len(dataset),2), 0, dtype=float)
    states[:,0]=dataset[:,0] #(price, 0)
    q_vals=dataset[:,2] #q_vals > thresh -> buy

    #fill actions categories
    actions_cat=np.full((len(dataset),n_actions), 0, dtype=int) #(hold, buy/sell)
    for i in range(len(dataset)):
        #convert actions to category
        action = [0,1] if q_vals[i] > q_threshold else [1,0]
        actions_cat[i]=action

    #generate train and test dataset
    N_test=int(len(dataset)*0.1)
    X_train_buf_coin=states[:-N_test]
    X_test_buf_coin=states[-N_test:]
    #Y_train_q_coin=q_vals[:-N_test]
    #Y_test_q_coin=q_vals[-N_test:]
    Y_train_cat_coin=actions_cat[:-N_test]
    Y_test_cat_coin=actions_cat[-N_test:]

    #add random buffers to coin buffer list 
    X_train_buf = catenate(X_train_buf, X_train_buf_coin)
    X_test_buf = catenate(X_test_buf, X_test_buf_coin)
    #Y_train_q = catenate(Y_train_q, Y_train_q_coin)
    #Y_test_q = catenate(Y_test_q, Y_test_q_coin)
    Y_train_cat = catenate(Y_train_cat, Y_train_cat_coin)
    Y_test_cat = catenate(Y_test_cat, Y_test_cat_coin)

#scale states in train set
X_train_buf=np.reshape(X_train_buf,(-1,2,1))
for i in range(len(X_train_buf)):
    scaler=scaler_list_train[i]
    X_train_buf[i]=scaler.transform(X_train_buf[i])
X_train_buf=np.reshape(X_train_buf,(-1,2))
#scale states in test set
X_test_buf=np.reshape(X_test_buf,(-1,2,1))
for i in range(len(X_test_buf)):
    scaler=scaler_list_test[i]
    X_test_buf[i]=scaler.transform(X_test_buf[i])
X_test_buf=np.reshape(X_test_buf,(-1,2))

#Clip scaled X dataset
X_train_clip=X_train.clip(min=0, max=1)
X_test_clip=X_test.clip(min=0, max=1)



#########################
#      NETWORK          #
#########################
print(colored('\nBuild Network','cyan'))
batch_size = None
#input_dim = 1
#output_dim = n_fut
#timesteps = n_past

#AGENT NETWORK
#inputs
input_fut_past = Input(batch_shape=(None, n_fut+n_past_out, 1))
input_cat = Input(batch_shape=(None, n_cat))

#Conv1D (q-net)
output_fut_past = Conv1D(filters=1024, kernel_size=2, activation=LeakyReLU(alpha=0.1))(input_fut_past)
output_fut_past=MaxPooling1D(pool_size=2)(output_fut_past)
output_fut_past = Conv1D(filters=512, kernel_size=4, activation=LeakyReLU(alpha=0.1))(output_fut_past)
output_fut_past=MaxPooling1D(pool_size=2)(output_fut_past)
output_fut_past = Conv1D(filters=256, kernel_size=4, activation=LeakyReLU(alpha=0.1))(output_fut_past)
output_fut_past=MaxPooling1D(pool_size=2)(output_fut_past)
output_fut_past = Conv1D(filters=128, kernel_size=4, activation=LeakyReLU(alpha=0.1))(output_fut_past)
output_fut_past=MaxPooling1D(pool_size=2)(output_fut_past)
output_fut_past = Flatten()(output_fut_past)
#combine the branches
output_layer = concatenate([output_fut_past, input_cat])#, input_state])
output_layer = Dense(512, activation=LeakyReLU(alpha=0.1))(output_layer)
output_layer = Dense(256, activation=LeakyReLU(alpha=0.1))(output_layer)
#output (q-net)
#output_q = Dense(1, activation='linear', name='pred')(output_layer)
output_action = Dense(n_actions, activation='softmax', name='cat')(output_layer)

#define model
model = Model(inputs=[input_fut_past, input_cat], outputs=output_action)
opt=Adam(lr = 0.001)
model.compile(optimizer=opt, loss={'cat':'categorical_crossentropy'}, 
              metrics={'cat':'accuracy'})

print(model.summary())
#plot_model(model, to_file="model.png", show_shapes=True)

#########################
#        TRAIN          #
#########################
print(colored('\n\nLoad network weights','cyan'))
#load saved weights
weight_folder="./"
#if not os.path.exists(weight_folder):
#    os.makedirs(weight_folder)
weight_file=weight_folder+"weight_dqn_"+net_type+"_"+timename+".h5"
try:
   model.load_weights(weight_file)
   print(colored('loaded weight file','green'))
except:
   print(colored('weight file not found','red'))

if len(sys.argv)>1 and sys.argv[1] == 'train':
    print(colored('\nTrain neural network','cyan'))
    #parameters
    epoch=100
    batch_size=32
    initial_lr=0.001
    final_lr=0.00001
    final_lr=0.000005
    decay_rate=0.9
    #exponential decaying learning rate
    step_epoch=round(len(X_train)/batch_size)
    step_tot=step_epoch*epoch
    ds_rate=math.log(decay_rate)/math.log(final_lr/initial_lr)
    decay_steps=step_tot*ds_rate
    lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate)
    #optimizer
    opt=Adam(learning_rate = lr_schedule)
    #callbacks (early stopping, model checkpoint)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30, restore_best_weights=True)
    cp = ModelCheckpoint(weight_file, monitor='val_accuracy', mode='max', save_weights_only=True, save_best_only=True, save_freq='epoch')
    lr_callback = LambdaCallback( on_epoch_begin= lambda epoch,logs: print("Optimizer step: %d" % (K.eval(model.optimizer.iterations))))
    #compile model
    model.compile(optimizer=opt, loss={'cat':'categorical_crossentropy'}, 
                  metrics={'cat':'accuracy'})
    #training
    model.fit(x=[X_train_clip, X_train_cat], y=Y_train_cat, batch_size=batch_size, epochs=epoch, validation_data=([X_test_clip, X_test_cat], Y_test_cat), callbacks=[es, cp, lr_callback])
    #save learned weights
    #model.save_weights(weight_file)


#########################
#      EVALUATE         #
#########################
if len(sys.argv)>1 and (sys.argv[1] == 'test' or sys.argv[1] == 'train'):
    print(colored('\nEvaluate network','cyan'))
    #prediction
    X=X_test
    #Y=Y_test_all
    #Y_cat=Y_test_cat
    prediction=model.predict([X_test_clip, X_test_cat], verbose=1)

    print(colored('\nUnnormalize data','cyan'))
    X_unnormalized=np.zeros(X_test.shape)
    #X_buf_unnormalized=X_test_buf_all
    X_buf_unnormalized=np.reshape(X_test_buf,(-1,2,1))
    for i in range(len(prediction)):
        scaler=scaler_list_test[i]
        X_unnormalized[i]=scaler.inverse_transform(X_test[i])
        X_buf_unnormalized[i]=scaler.inverse_transform(X_buf_unnormalized[i])
        print(f"Step: {i}/{len(prediction)}\r", end='')
    X_unnormalized=np.reshape(X_unnormalized,(-1,n_fut+n_past_out))
    X_buf_unnormalized=np.reshape(X_buf_unnormalized,(-1,2))

    print(colored('\n\nSave results','cyan'))
    result=np.zeros(shape=(len(prediction), n_fut+n_past_out+n_state+2*n_actions+2))
    result[:,:n_fut+n_past_out]=X_unnormalized
    result[:,n_fut+n_past_out:n_fut+n_past_out+n_state]=X_buf_unnormalized
    result[:,-3*n_actions:-2*n_actions]=prediction
    result[:,-2*n_actions:-n_actions]=Y_test_cat
    #result[:,-2]=prediction
    #result[:,-1]=Y_test_q
    save_folder='./'
    save_file=save_folder+'output_dataset_'+net_type+'_'+timename+'.csv'
    print("Saving to "+save_file)
    with open(save_file, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(result)):
            print(f"Step: {i}/{len(result)}\r", end='')
            line=result[i]
            wr.writerow(line)


print(colored('\n\nFinish','cyan'))
