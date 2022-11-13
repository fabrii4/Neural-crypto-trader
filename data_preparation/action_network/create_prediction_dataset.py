from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dropout, Lambda, concatenate
from tensorflow.keras.optimizers import SGD, Adam
#from tensorflow.keras.utils.vis_utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from tcn import TCN, tcn_full_summary
import numpy as np
#from sklearn.model_selection import train_test_split
from termcolor import colored
from itertools import product
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

#########################
#      NETWORK          #
#########################
print(colored('\nBuild Network','cyan'))
# define the keras layers
batch_size, timesteps, input_dim, output_dim, output_class = None, 256, 16, 16, 5
#inputs
input_layer = Input(batch_shape=(batch_size, timesteps, input_dim))

#outputs
output_layer0 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=True, activation='relu')(input_layer)
output_layer0 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=False, activation='relu')(output_layer0)

output_layer1 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=True, activation='relu')(input_layer)
output_layer1 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=False, activation='relu')(output_layer1)

output_layer0 = Dense(512, activation='relu')(output_layer0)
output_layer0 = Dense(256, activation='relu')(output_layer0)
output_layer0 = Dense(output_dim, activation='linear', name='pred')(output_layer0)

output_layer1 = Dense(512, activation='relu')(output_layer1)
output_layer1 = Dense(256, activation='relu')(output_layer1)
output_layer1 = Dense(output_class, activation='softmax', name='cat')(output_layer1)

#define model
model = Model(inputs=[input_layer], outputs=[output_layer0, output_layer1])
opt=Adam(lr = 0.001)
model.compile(optimizer=opt, loss={'pred':'mse','cat':'categorical_crossentropy'}, 
              metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})

#@TODO use load architecture
#@TODO use less trained weight file

##Load model architecture
#architecture_file="../../prediction_network/training/architecture.json"
#with open(architecture_file, 'r') as json_file:
#    json_config = json_file.read()
#model = model_from_json(json_config)

# plot model
#tcn_full_summary(model, expand_residual_blocks=False)
print(model.summary())
#plot_model(model, to_file="model.png", show_shapes=True)



#########################
#      DATASET          #
#########################
#Paths
coins=['BTCEUR', 'BTCUSDT', 'LTCEUR', 'LTCUSDT', 'XRPEUR', 'XRPUSDT', 'ETHEUR', 'ETHUSDT', 'DOGEEUR', 'DOGEUSDT', 'ADAEUR', 'ADAUSDT', 'XLMEUR', 'XLMUSDT', 'XMRUSDT', 'SOLEUR', 'SOLUSDT', 'XTZUSDT']
#coins=['DOGEEUR']

#price variation for categories classification dict (depending on timestep)
cat_range_dict={'4h': [0.30, 0.10], '1h': [0.15, 0.05], '5m': [0.0125, 0.00416667]}
#timenames=['4h', '1h', '5m']
timenames=['4h', '1h']

for coin, timename in product(coins,timenames):
    data_folder="../../data/"

    coin_folder=data_folder+coin+"/tech_sent/"
    #timename='1h'

    print(colored('\nLoad Dataset','cyan'))
    #if len(sys.argv)>2: 
    #    timename=sys.argv[2]
    filename=coin+"_"+timename+"_tsi.csv"
    print("Loading file: "+filename)
    #import dataset file
    dataset = np.genfromtxt(coin_folder+filename, delimiter=',')
    #delete header
    dataset = np.delete(dataset, (0), axis=0)
    #for '5m' timestep reduce size of loaded dataset
    if timename=='5m':
        N_data=int(len(dataset)/12)
        dataset=dataset[-N_data:]
    #features_ind
    data_ind=[4,5] #close price, volume
    tech_ind=[12,13,14,15,16,17,18,19,20] #macd(3), roc, sma, rsi, cci, ema, atr
    #sent_ind=[21,22,23,24,25,26,27,28,29] #rsent, rvol, rsma(2),tsent,tvol,tsma(2),fgi
    sent_ind=[21,22,23,24,25] #tsent, tvol, tsma(2), fgi
    features_ind = data_ind + tech_ind + sent_ind
    
    ind_out=features_ind.index(4)

    data_list=[]
    #extract feautures
    for ind in features_ind:
        # import the dataset
        data = np.flip(dataset[:,ind])
        data = np.reshape(data,(-1,1))
        data = np.nan_to_num(data)
        #data = data[:50000]
        data_list.append(data)
    data_list=np.array(data_list)
    #reshape (_, input_dim)
    data_list=np.transpose(data_list,(1,0,2))
    data_list=np.reshape(data_list,(-1,input_dim))

    #time list
    time_list=np.flip(dataset[:,0])
    time_list=time_list[output_dim:-timesteps+1]

    print(colored('\nNormalize and prepare data','cyan'))
    delta=1 #step between training elements
    if len(sys.argv)>3: 
        delta=int(sys.argv[3])
    ind=output_dim
    X=[]
    Y=[]
    Y_cat=[] #large decrease, small decrease, constant, small increase, large increase
    scaler_list=[]
    while(True):
        cat_range=cat_range_dict[timename]
        #arrange data in past and future samples
        sample_past=data_list[ind:ind+timesteps].copy()
        sample_fut=data_list[ind-output_dim:ind,ind_out].copy()
        #create trend categories before normalizing
        Y_cat_sample=[0,0,1,0,0] #constant
        present_val=sample_past[0, ind_out]
        del_max=max(sample_fut)-present_val
        del_min=present_val-min(sample_fut)
        if del_max > del_min:
            if del_max >= present_val * cat_range[0]: #>15% increase
                Y_cat_sample=[0,0,0,0,1]
            elif del_max >= present_val * cat_range[1]: #>5% increase
                Y_cat_sample=[0,0,0,1,0]
        else:
            if del_min >= present_val * cat_range[0]: #>15% decrease
                Y_cat_sample=[1,0,0,0,0]
            elif del_min >= present_val * cat_range[1]: #>5% decrease
                Y_cat_sample=[0,1,0,0,0]
        #normalize past and future
        sample_past=np.reshape(sample_past,(-1,input_dim,1))
        sample_fut=np.reshape(sample_fut,(-1,1))
        for i in range(len(features_ind)):
            scaler = MinMaxScaler((0.33,0.66))
            sample_past[:,i]=scaler.fit_transform(sample_past[:,i])
            if i == ind_out:
                sample_fut=scaler.transform(sample_fut)
                scaler_list.append(scaler)
        sample_past=np.reshape(sample_past,(-1,input_dim))
        sample_fut=np.reshape(sample_fut,-1)
        #populate X and Y datasets
        Y_sample=sample_fut
        X_sample=sample_past
        Y.append(Y_sample)
        X.append(X_sample)
        Y_cat.append(Y_cat_sample)
        ind+=delta
        print(f"Step: {ind}/{len(data_list)-timesteps}\r", end='')
        if ind+timesteps > len(data_list):
            break
    X=np.array(X)
    Y=np.array(Y)
    Y_cat=np.array(Y_cat)
    scaler_list=np.array(scaler_list)

    #add coin dataset to total dataset
    X_train = X
    Y_train = Y
    Y_train_cat = Y_cat


    #########################
    #        LOAD           #
    #########################
    print(colored('\n\nLoad network weights','cyan'))
    #load saved weights
    weight_folder="../../prediction_network/training/weights/"
    #if not os.path.exists(weight_folder):
    #    os.makedirs(weight_folder)
    weight_file=weight_folder+"weight_tcn_action_"+timename+".h5"
    print(weight_file)
    try:
       model.load_weights(weight_file)
       print(colored('loaded weight file','green'))
    except:
       print(colored('weight file not found','red'))

    #Evaluate final model
    model.compile(optimizer='adam', loss={'pred':'mse','cat':'categorical_crossentropy'}, 
                      metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})
    model.evaluate(X_train, [Y_train, Y_train_cat])


    #########################
    #      EVALUATE         #
    #########################
    #if len(sys.argv)>1 and sys.argv[1] == 'test':
    if True:
        print(colored('\nEvaluate network','cyan'))
        #prediction
        X=X_train
        Y=Y_train
        Y_cat=Y_train_cat
        prediction_tot=model.predict(X, verbose=1)
        #prediction=reshape(prediction,(-1,output_dim))
        prediction=prediction_tot[0]
        prediction_cat=prediction_tot[1]

        print(colored('\nUnnormalize data','cyan'))
        X=X[:,:,ind_out]
        prediction_unnormalized=np.zeros(prediction.shape)
        X_unnormalized=np.zeros(X.shape)
        Y_unnormalized=np.zeros(Y.shape)
        for i in range(len(prediction)):
            prediction_unnormalized[i]=np.reshape(
                scaler_list[i].inverse_transform(np.reshape(prediction[i],(-1,1))),-1)
            X_unnormalized[i]=np.reshape(scaler_list[i].inverse_transform(np.reshape(X[i],(-1,1))),-1)
            Y_unnormalized[i]=np.reshape(scaler_list[i].inverse_transform(np.reshape(Y[i],(-1,1))),-1)
            print(f"Step: {i}/{len(prediction)}\r", end='')
        #X_test=X_test[:,:,ind_out]

        print(colored('\n\nSave results','cyan'))
        result=np.zeros(shape=(len(prediction), 2*output_dim+timesteps+2*output_class+1))
        #result[:,0]=X_time_list[i]
        result[:,:output_dim]=Y_unnormalized
        result[:,output_dim:2*output_dim]=prediction_unnormalized
        result[:,2*output_dim:2*output_dim+timesteps]=X_unnormalized[:,:timesteps]
        result[:,2*output_dim+timesteps:2*output_dim+timesteps+output_class]=prediction_cat
        result[:,2*output_dim+timesteps+output_class:-1]=Y_cat
        result[:,-1]=time_list
        #save path
        save_folder=data_folder+coin+"/action_training/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_file=save_folder+'prediction_dataset_'+timename+"_"+coin+'.csv'
        print("Saving to "+save_file)
        with open(save_file, 'w', newline='') as f:
            wr = csv.writer(f)
            for i in range(len(result)):
                print(f"Step: {i}/{len(result)}\r", end='')
                line=result[i]
                wr.writerow(line)

print(colored('\n\nFinish','cyan'))
