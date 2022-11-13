from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dropout, Lambda, concatenate, LeakyReLU
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

#Save model architecture
json_config = model.to_json()
with open ('architecture.json', 'w') as f:
    json.dump(json_config, f)
#Load model architecture
#json_config = json.load('architecture.json')
#model = model_from_json(json_config)


# plot model
#tcn_full_summary(model, expand_residual_blocks=False)
print(model.summary())
#plot_model(model, to_file="model.png", show_shapes=True)



#########################
#      DATASET          #
#########################
#Paths
coins=['BTCUSDT', 'LTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'ADAUSDT', 'XLMUSDT', 'XMRUSDT', 'BTCEUR', 'LTCEUR', 'XRPEUR', 'ETHEUR', 'DOGEEUR', 'ADAEUR', 'XLMEUR']
#coins=['BTCUSDT', 'LTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'ADAUSDT', 'XLMUSDT', 'XMRUSDT']
#coins=['BTCUSDT', 'LTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'ADAUSDT', 'BTCEUR', 'LTCEUR', 'XRPEUR', 'ETHEUR', 'DOGEEUR', 'ADAEUR']
#coins=['DOGEEUR', 'LTCEUR']
#coins=['XRPUSDT']

#price variation for categories classification dict (depending on timestep)
cat_range_dict={'4h': [0.30, 0.10], '1h': [0.15, 0.05], '5m': [0.0125, 0.00416667]}
#portion of dataset to use
data_len_scale={'4h':1, '1h':1/2, '5m':1/12}
#training time scale
timename='1h'

X_train=np.array([])
Y_train=np.array([])
Y_train_cat=np.array([])
X_test=np.array([])
Y_test=np.array([])
Y_test_cat=np.array([])
scaler_list=np.array([])

for n in range(len(coins)):
    data_folder="../../data/"
    folder=data_folder+coins[n]+"/tech_sent/"
    coin=coins[n]+"_"

    print(colored('\nLoad Dataset','cyan'))
    #if len(sys.argv)>2: 
    #    timename=sys.argv[2]
    filename=coin+timename+"_tsi.csv"
    print("Loading file: "+filename)
    #import dataset file
    dataset = np.genfromtxt(folder+filename, delimiter=',')
    #delete header
    dataset = np.delete(dataset, (0), axis=0)
    #scale size of loaded dataset
    N_data=int(len(dataset)*data_len_scale[timename])
    dataset=dataset[-N_data:]
    #features_ind
    data_ind=[4,5] #close price, volume
    tech_ind=[12,13,14,15,16,17,18,19,20] #macd(3), roc, sma, rsi, cci, ema, atr
    #sent_ind=[21,22,23,24,25,26,27,28,29] #rsent, rvol, rsma(2), tsent, tvol, tsma(2), fgi
    sent_ind=[21,22,23,24,25] #tsent, tvol, tsma(2), fgi
    #lunar_ind=[26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]
    #lunar_ind=[31,32,33,34,35,42] #tweet sent (Very Bearish, Bearish, Neutral, Bullish, 
                                  #Very Bullish), number of news
    #lunar_ind=[26,27,28,29,30,31,32,33,34] #n tweet, mean sent, galaxy, (sma100, sma16)*3
    features_ind = data_ind + tech_ind + sent_ind# + lunar_ind
    
    #prediction is based on close price
    ind_out=features_ind.index(4)

    #ind mean sent (already normalized)
    #ind_sent=[features_ind.index(27),features_ind.index(31),features_ind.index(32)]

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

    #r_out ratio of latest elements dropped from dataset for cross validation training
    if len(sys.argv)>2:
        r_out=float(sys.argv[2])
        N_out=int(len(data_list)*r_out)
        data_list=data_list[N_out:]

    print(colored('\nNormalize and prepare data','cyan'))
    delta=1 #step between training elements
    if len(sys.argv)>3: 
        delta=int(sys.argv[3])
    ind=output_dim
    X=[]
    Y=[]
    Y_cat=[] #large decrease, small decrease, constant, small increase, large increase
    scaler_list_part=[]
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
            #if i in ind_sent:
            #    #mean sent are already normalized
            #    continue
            scaler = MinMaxScaler((0.33,0.66))
            sample_past[:,i]=scaler.fit_transform(sample_past[:,i])
            if i == ind_out:
                sample_fut=scaler.transform(sample_fut)
                scaler_list_part.append(scaler)
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
    scaler_list_part=np.array(scaler_list_part)

    #split coin dataset
    N_test=int(len(X)*0.1)
    #the final training is done on the entire dataset
    if len(sys.argv)>1 and (sys.argv[1] == 'train_final' or sys.argv[1] == 'train_action'):
        N_test=0
    X_train_part=X[N_test:]
    X_test_part=X[:N_test]
    Y_train_part=Y[N_test:]
    Y_test_part=Y[:N_test]
    Y_train_cat_part=Y_cat[N_test:]
    Y_test_cat_part=Y_cat[:N_test]
    scaler_list_test=scaler_list_part[:N_test]
    #add coin dataset to total dataset
    X_train = X_train_part if X_train.size == 0 else np.concatenate((X_train, X_train_part))
    Y_train = Y_train_part if Y_train.size == 0 else np.concatenate((Y_train, Y_train_part))
    Y_train_cat = Y_train_cat_part if Y_train_cat.size == 0 else np.concatenate((Y_train_cat, Y_train_cat_part))
    X_test = X_test_part if X_test.size == 0 else np.concatenate((X_test, X_test_part))
    Y_test = Y_test_part if Y_test.size == 0 else np.concatenate((Y_test, Y_test_part))
    Y_test_cat = Y_test_cat_part if Y_test_cat.size == 0 else np.concatenate((Y_test_cat, Y_test_cat_part))
    scaler_list = scaler_list_test if scaler_list.size == 0 else np.concatenate((scaler_list, scaler_list_test))


#########################
#        TRAIN          #
#########################
print(colored('\n\nLoad network weights','cyan'))
#load saved weights
weight_folder="./"
weight_folder_action="./weights/"
if not os.path.exists(weight_folder_action):
    os.makedirs(weight_folder_action)
train_step="_"+str(sys.argv[2]) if len(sys.argv)>2 else ""
weight_file=weight_folder+"weight_tcn_"+timename+"_tsc.h5"
weight_file_final=weight_folder+"weight_tcn_"+timename+"_tsc_final.h5"
if sys.argv[1] == 'train_action':
    weight_file=""
print(weight_file)
try:
   model.load_weights(weight_file)
   print(colored('loaded weight file','green'))
except:
   print(colored('weight file not found','red'))

if len(sys.argv)>1 and sys.argv[1] == 'train':
    print(colored('\nTrain neural network','cyan'))
    epoch=50
    batch_size=25
    initial_lr=0.001
    final_lr=0.00001
    decay_rate=0.9
    step_epoch=round(len(X_train)/batch_size)
    step_tot=step_epoch*epoch
    ds_rate=math.log(decay_rate)/math.log(final_lr/initial_lr)
    decay_steps=step_tot*ds_rate
    lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate)
    opt=Adam(learning_rate = lr_schedule)
    model.compile(optimizer=opt, loss={'pred':'mse','cat':'categorical_crossentropy'}, 
                  metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})
    es = EarlyStopping(monitor='val_cat_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=True)
    cp = ModelCheckpoint(weight_file, monitor='val_cat_accuracy', mode='max', save_weights_only=True, save_best_only=True, save_freq='epoch')
    lr_callback = LambdaCallback( on_epoch_begin= lambda epoch,logs: print("Optimizer step: %d" % (K.eval(model.optimizer.iterations))))
    #training
    model.fit(X_train, [Y_train, Y_train_cat], batch_size=batch_size, epochs=epoch, validation_data=(X_test, [Y_test, Y_test_cat]), callbacks=[es, cp, lr_callback])




#final training step without validation (used also to train weights for action dataset)
if len(sys.argv)>1 and (sys.argv[1] == 'train_final' or sys.argv[1] == 'train_action'):
    print(colored('\nFinal training without validation','cyan'))
    epoch=6
    batch_size=25
    initial_lr=0.0001
    final_lr=0.000005
    decay_rate=0.9
    if sys.argv[1] == 'train_action':
        initial_lr=0.001
        epoch=20
    step_epoch=round(len(X_train)/batch_size)
    step_tot=step_epoch*epoch
    ds_rate=math.log(decay_rate)/math.log(final_lr/initial_lr)
    decay_steps=step_tot*ds_rate
    lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate)
    opt=Adam(learning_rate = lr_schedule)
    model.compile(optimizer=opt, loss={'pred':'mse','cat':'categorical_crossentropy'}, 
                      metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})
    cp = ModelCheckpoint(weight_file_final, save_weights_only=True, save_best_only=False, save_freq='epoch')
    #used for creating weight files for training action network
    cp_action = ModelCheckpoint(weight_folder_action+"weight_tcn_{epoch:02d}-{pred_loss:.4f}-{cat_accuracy:.2f}.h5", save_weights_only=True, save_best_only=False, save_freq='epoch')
    if sys.argv[1] == 'train_action':
        #remove existing weights before training for action dataset
        for f in os.listdir(weight_folder_action):
            os.remove(os.path.join(weight_folder_action, f))
        cp=cp_action
    model.fit(X_train, [Y_train, Y_train_cat], batch_size=batch_size, epochs=epoch, callbacks=[cp])
    if sys.argv[1] == 'train_action':
        print("Finish")
        exit()
    model.save_weights(weight_file_final)





#Evaluate final model
if len(sys.argv)>1 and (sys.argv[1] == 'test' or sys.argv[1] == 'train'):
    print(colored('\n\nEvaluate model','cyan'))
    try:
       model.load_weights(weight_file)
       print(colored('loaded weight file','green'))
    except:
       print(colored('weight file not found','red'))
    model.compile(optimizer='adam', loss={'pred':'mse','cat':'categorical_crossentropy'}, 
                      metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})
    model.evaluate(X_test, [Y_test, Y_test_cat])

##Evaluate final model
#print(colored('\n\nEvaluate model','cyan'))
#try:
#   model.load_weights(weight_file_final)
#   print(colored('loaded weight file','green'))
#except:
#   print(colored('weight file not found','red'))
#model.compile(optimizer='adam', loss={'pred':'mse','cat':'categorical_crossentropy'}, 
#                  metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})
#model.evaluate(X_test, [Y_test, Y_test_cat])


#########################
#      EVALUATE         #
#########################
if len(sys.argv)>1 and (sys.argv[1] == 'test' or sys.argv[1] == 'train'):
    print(colored('\nEvaluate network','cyan'))
    #prediction
    X=X_test
    Y=Y_test
    Y_cat=Y_test_cat
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
    result=np.zeros(shape=(len(prediction), 2*output_dim+timesteps+2*output_class))
    #result[:,0]=X_time_list[i]
    result[:,:output_dim]=Y_unnormalized
    result[:,output_dim:2*output_dim]=prediction_unnormalized
    result[:,2*output_dim:2*output_dim+timesteps]=X_unnormalized[:,:timesteps]
    result[:,2*output_dim+timesteps:2*output_dim+timesteps+output_class]=prediction_cat
    result[:,2*output_dim+timesteps+output_class:]=Y_cat
    save_folder='./'
    save_file=save_folder+'output_dataset_'+timename+train_step+'.csv'
    print("Saving to "+save_file)
    with open(save_file, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(result)):
            print(f"Step: {i}/{len(result)}\r", end='')
            line=result[i]
            wr.writerow(line)

print(colored('\n\nFinish','cyan'))
