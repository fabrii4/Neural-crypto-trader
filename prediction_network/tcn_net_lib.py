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

#do not use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings
warnings.filterwarnings("ignore")

#Paths
coin_folder="../data/LTCUSDT/"
folder=coin_folder+"tech_ind/"
coin="LTCUSDT_"
timename='1h'

save_folder=coin_folder+"output_datasets/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#########################
#      NETWORK          #
#########################
def load_model(data):
    timename=data.timestep
    # define the keras layers
    batch_size=None
    timesteps=data.length #256
    input_dim=len(data.data) #16
    output_dim=data.pred_length #16
    output_class=data.cat_length #5
    #inputs
    input_layer = Input(batch_shape=(batch_size, timesteps, input_dim))
    #output trend prediction
    output_layer0 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=True, activation='relu')(input_layer)
    output_layer0 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=False, activation='relu')(output_layer0)
    output_layer0 = Dense(512, activation='relu')(output_layer0)
    output_layer0 = Dense(256, activation='relu')(output_layer0)
    output_layer0 = Dense(output_dim, activation='linear', name='pred')(output_layer0)
    #output class prediction
    output_layer1 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=True, activation='relu')(input_layer)
    output_layer1 = TCN(nb_filters=128, kernel_size=8, nb_stacks=2, dilations=[1,2,4,8,16], padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=False, activation='relu')(output_layer1)
    output_layer1 = Dense(512, activation='relu')(output_layer1)
    output_layer1 = Dense(256, activation='relu')(output_layer1)
    output_layer1 = Dense(output_class, activation='softmax', name='cat')(output_layer1)
    #define model
    model = Model(inputs=[input_layer], outputs=[output_layer0, output_layer1])
    opt=Adam(lr = 0.001)
    model.compile(optimizer=opt, loss={'pred':'mse','cat':'categorical_crossentropy'}, 
                  metrics={'pred': 'mean_absolute_error', 'cat':'accuracy'})

    #load weights
    weight_folder="../prediction_network/weights/"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    weight_file=weight_folder+"weight_tcn_"+timename+"_tsc.h5"
    try:
       model.load_weights(weight_file)
       print(colored('loaded prediction weight file','green'))
    except:
       print(colored('prediction weight file not found','red'))

    return model
    # plot model
    #tcn_full_summary(model, expand_residual_blocks=False)
    #print(model.summary())
    #plot_model(model, to_file="model.png", show_shapes=True)

def predict(model, data):
    prediction_tot = model.predict(data.norm_data)
    prediction=prediction_tot[0]
    prediction_cat=prediction_tot[1]
    prediction = np.transpose(prediction)
    prediction = data.scaler[0].inverse_transform(prediction)
    #prediction = np.flip(prediction)
    return prediction, prediction_cat

