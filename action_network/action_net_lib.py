import numpy as np
import pandas as pd
import gym
import os
import sys
import random
from termcolor import colored

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

sys.path.insert(1, '../optimal_buy-sell/')
import buy_sell_lib as bs

#do not use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#scaler that keeps present at constant position p01 and does not deform the absolute scale variation of the data (this should be usefult to determine if a variation is above treshold and should be considered)
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


def load_model(data):
    timename=data.timestep
    # define the keras layers
    batch_size = None
    input_dim = 1
    n_fut = data.pred_length #16
    n_past_out = int(data.length/2) #128
    n_cat = data.cat_length
    n_actions = 2

    #AGENT NETWORK
    #inputs
    input_fut_past = Input(batch_shape=(None, n_fut+n_past_out, 1))
    input_cat = Input(batch_shape=(None, n_cat))
    #Conv1D
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
    output_layer = concatenate([output_fut_past, input_cat])
    output_layer = Dense(512, activation=LeakyReLU(alpha=0.1))(output_layer)
    output_layer = Dense(256, activation=LeakyReLU(alpha=0.1))(output_layer)
    #output actions
    output_action = Dense(n_actions, activation='softmax')(output_layer)
    #create buy and sell models
    model_buy = Model(inputs=[input_fut_past, input_cat], outputs=output_action)
    model_buy.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model_sell = Model(inputs=[input_fut_past, input_cat], outputs=output_action)
    model_sell.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    #load weights
    weight_folder="../action_network/weights/"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    weight_buy=weight_folder+"weight_dqn_buy_"+timename+".h5"
    weight_sell=weight_folder+"weight_dqn_sell_"+timename+".h5"
    try:
        model_buy.load_weights(weight_buy)
        print(colored('loaded buy weight file','green'))
    except:
        print(colored('buy weight file not found','red'))
    try:
        model_sell.load_weights(weight_sell)
        print(colored('loaded sell weight file','green'))
    except:
        print(colored('sell weight file not found','red'))

    return model_buy, model_sell


ind_close=0
def prepare_state(data):
    n_past_out = int(data.length/2) #128
    n_fut=data.pred_length
    past=data.norm_data[0,:n_past_out,ind_close].copy()
    past = np.reshape(past,(-1,1))
    #unnormalize past data
    past = data.scaler[ind_close].inverse_transform(past)
    #add current price to past data
    if data.curr_price != past[0,0]:
        past=np.insert(past,0,[data.curr_price], axis=0)
        past=past[:n_past_out]
    future = data.prediction.copy()
    state_fut_past = np.concatenate((future, past))
    state_fut_past=np.reshape(state_fut_past, (1,-1,1))
    #normalize data using action scaler instead of MinMax
    scaler=Scaler(state_fut_past[0][n_fut][0],0.5,0.05,0.5)
    state_fut_past[0]=scaler.transform(state_fut_past[0])
    #Clip normalized data
    state_fut_past[0]=state_fut_past[0].clip(min=0, max=1)
    state_cat=np.round(data.prediction_cat)
    return state_fut_past, state_cat



def action(model_buy, model_sell, data):
    commission=data.commission
    total_invested=data.coin_data.total_invested
    coin_held=data.coin_data.coin_held
    buy_price=total_invested/coin_held if coin_held>0 else 0
    current_price=data.curr_price
    state_fut_past, state_cat = prepare_state(data)
    fut=data.prediction.copy()

    prediction=None
    action=[1,0,0]
    if buy_price > 0: 
        #sell
        if current_price > buy_price*(1+commission):
            prediction=model_sell.predict([state_fut_past, state_cat])
            pred_act=prediction[0]
            if pred_act[1]>pred_act[0] and max(fut) <= current_price*(1+commission):
                action=[0,0,1]
        #sell in case price start to drop
        if current_price <= buy_price*(1+commission):
            if max(fut) <= current_price and min(fut) < current_price*(1-commission):
                if data.wait_to_sell>0:
                    action=[0,0,1]
                    data.wait_to_sell=0
                else:
                    data.wait_to_sell+=1
    else:
        #buy
        prediction=model_buy.predict([state_fut_past, state_cat])
        pred_act=prediction[0]
        if pred_act[1]>pred_act[0] and min(fut) >= current_price*(1-commission):
            action=[0,1,0]


    action = np.argmax(action)
    return action

#action using q-algorithm
def action_q(model_buy, model_sell, data):
    commission=data.commission
    gamma=0.9
    buy_tresh=0.1
    def q(price, buy_price, commission=commission):
        return (price-buy_price)/buy_price*(1-commission)
    total_invested=data.coin_data.total_invested
    coin_held=data.coin_data.coin_held
    buy_price=total_invested/coin_held if coin_held>0 else 0
    current_price=data.curr_price
    state_fut_past, state_cat = prepare_state(data)
    fut=data.prediction.copy()

    action=0
    fut=np.flip(fut, axis=0)
    #sell
    if buy_price>0:
        q_pres=q(current_price,buy_price)
        q_fut=[q(fut[t],buy_price) for t in range(len(fut))]
        neg_pos=[ind for ind, el in enumerate(q_fut) if el<0]
        if len(neg_pos)>0:
            q_fut[neg_pos[0]:]=[0 for t in range(neg_pos[0],len(q_fut))]
        q_fut=max(q_fut)
        if q_pres > 0 and q_pres > q_fut:
            action=2
    #buy
    else:
        q_pres=[q(fut[t],current_price) for t in range(len(fut))]
        q_fut=[[q(fut[t],fut[t1]) if t1 < t else 0 for t in range(len(fut))] for t1 in range(len(fut))]
        q_fut=np.array(q_fut)
        q_buy=[[q_pres[n],max(q_fut[:,n])] for n in range(len(fut))]
        q_buy=np.array(q_buy)
        #q=Sum[(q_i-dq_i*(HT(-qf_i)-HT(dq_i,qf_i))-qf_i*HT(q_i,qf_i,-dq_i))*gamma**i]
        q_buy_scalar=sum((q_buy[n,0]-(q_buy[n,0]-q_buy[n,1])*(np.heaviside(-q_buy[n,1],1)-np.heaviside(q_buy[n,0]-q_buy[n,1],1)*np.heaviside(q_buy[n,1],1))-q_buy[n,1]*np.heaviside(q_buy[n,0],1)*np.heaviside(q_buy[n,1],1)*np.heaviside(q_buy[n,1]-q_buy[n,0],1))*gamma**(n+1) for n in range(len(fut)))
        if q_buy_scalar > buy_tresh:
            action=1

    return action


def validate_action(action, data):
    is_valid=False
    #@TODO implement checkers
    #@TODO check websocket error High CPU usage
#    coin=data.coin
#    curr_timestamp=data.curr_timestamp
#    curr_price=data.curr_price
#    Dt=120*1000 #2min in millisec

#    commission=data.commission
#    total_invested=data.coin_data.total_invested
#    coin_held=data.coin_data.coin_held
#    buy_price=total_invested/coin_held if coin_held>0 else 0
#    buy_price=buy_price*(1+commission)

#    #buy
#    if action == 1:
#        is_valid = bs.buy(data, Dt)
#    #sell
#    if action == 2:
#        is_valid = bs.sell(data, Dt)

    is_valid=True

    return is_valid


#dummy action from environment
def take_action(data, action):
    is_executed=False

    if not validate_action(action, data):
        return data, is_executed

    #if action_type < 1:
    if action == 1:
        fraction = 1#(1+action)/n_class
        # Buy amount % of balance in coin
        total_possible = data.coin_data.balance
        amount = total_possible * fraction
        if amount > 0:
            coin_bought = amount / data.curr_price
            data.coin_data.balance =0#-= amount
            data.coin_data.coin_held += coin_bought
            data.coin_data.total_invested += amount
            is_executed=True

    #elif action_type < 2:
    elif action == 2:
        fraction = 1#(1+action-n_class)/n_class
        # Sell amount % of shares held
        coin_sold = data.coin_data.coin_held * fraction
        amount = coin_sold * data.curr_price * (1 - data.commission)
        if amount > 0:# and amount > self.total_invested * fraction * 1.002:
            data.coin_data.balance += amount
            data.coin_data.coin_held =0#-= coin_sold
            data.coin_data.total_invested = 0
            is_executed=True

    data.coin_data.net_worth = data.coin_data.balance + data.coin_data.total_invested

    return data, is_executed



def update_buy_sell_points(data, is_exec, action, buy, sell):
    if is_exec and action > 0:
        if action == 1:
            buy.append([data.curr_timestamp, data.curr_price])
        elif action == 2:
            sell.append([data.curr_timestamp, data.curr_price])
    buy=buy[-data.length:]
    sell=sell[-data.length:]

    return buy, sell
