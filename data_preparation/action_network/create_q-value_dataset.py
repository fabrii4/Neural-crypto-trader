import math
import numpy as np
import random as rn
import csv
import time
import os
import sys
from termcolor import colored


#Paths
coins=['BTCEUR', 'BTCUSDT', 'LTCEUR', 'LTCUSDT', 'XRPEUR', 'XRPUSDT', 'ETHEUR', 'ETHUSDT', 'DOGEEUR', 'DOGEUSDT', 'ADAEUR', 'ADAUSDT', 'XLMEUR', 'XLMUSDT', 'XMRUSDT', 'SOLEUR', 'SOLUSDT', 'XTZUSDT']
#coins=['DOGEEUR', 'LTCEUR']
#coins=['DOGEEUR']

data_folder="../../data/"
#timename='5m'
#timenames=['4h', '1h', '5m']
timenames=['4h', '1h']

#parameters
n_fut=16
n_past=128
n_past_out=128
n_cat=5
n_state=2
n_actions=3
commission=0.01
gamma=0.95
Nmax=100
buy_treshold=0.02

#buy now and sell in the past
def q_buy_past(p, dt, Nmax, commission):
    q = [[(p[max(t-(i+1),0)]-p[min(t+dt,len(p)-1)])/p[min(t+dt,len(p)-1)]*(1-commission) for i in range(Nmax)] for t in range(len(p))]
    return np.array(q)
#buy now and sell in the future
def q_buy_fut(p, dt, Nmax, commission):
    q = [[(p[min(t+(i+1),len(p)-1)]-p[min(t+dt,t+(i+1),len(p)-1)])/p[min(t+dt,t+(i+1),len(p)-1)]*(1-commission) for i in range(Nmax)] for t in range(len(p))]
    return np.array(q)
#sell now and buy in the past
def q_sell_past(p, dt, Nmax, commission):
    q = [[(p[min(t+dt,len(p)-1)]-p[max(t-(i+1),0)])/p[max(t-(i+1),0)]*(1-commission) for i in range(Nmax)] for t in range(len(p))]
    return np.array(q)
#sell now and buy in the future
def q_sell_fut(p, dt, Nmax, commission):
    q = [[(p[min(t+dt,t+(i+1),len(p)-1)]-p[min(t+(i+1),len(p)-1)])/p[min(t+(i+1),len(p)-1)]*(1-commission) for i in range(Nmax)] for t in range(len(p))]
    return np.array(q)

#Load time series datasets
#i_coin=0
#coin=coin_list[i_coin]
for coin in coins:
    for timename in timenames:
        print(colored('\nLoad Dataset','cyan'))
        #if len(sys.argv)>2: 
        #    timename=sys.argv[2]
        filename='prediction_dataset_'+timename+"_"+coin+'.csv'
        folder=data_folder+coin+"/action_training/"
        print("Loading file: "+filename)
        #state dataset (future trend prediction + past prices + category prediction)
        dataset = np.genfromtxt(folder+filename, delimiter=',')
        # flip dataset, time increases along axis 0 (vertical) and decreases along axis 1
        #...t+1, t_0, t-1....
        #...t+2, t+1, t_0,... 
        dataset = np.flip(dataset, axis=0)
        #get true future
        fut=np.flip(dataset[:,:n_fut], axis=1)
        #get current price dataset
        p=dataset[:,2*n_fut]
        #get current time
        t_list=dataset[:,-1]

        #future minimum
        p_fut_min=np.array([min(fut[i]) for i in range(len(fut))])    

        #remove future negative sell points
        for i in range(len(fut)):
            neg_pos=[ind for ind, el in enumerate(fut[i]) if el<p[i]*(1-commission)]
            if len(neg_pos)>0:
                fut[i,neg_pos[0]:]=[0 for n in range(neg_pos[0], len(fut[i]))]
        p_fut=np.array([max(fut[i]) for i in range(len(fut))])


        #create final results (p,p_fut_max,qsell)
        result_sell=[[p[t],p_fut[t], int(p[t]>p_fut[t]), t_list[t]] for t in range(len(p))]
        #result_buy=[[p[t],p_fut[t], int(p_fut[t]>p[t]*(1+commission) and p_fut_min[t]>p[t]*(1-commission) and fut[t,0]>p[t] and fut[t,1]>p[t])] for t in range(len(p))]

        result_buy=[[p[t],p_fut[t], int(p_fut[t]>(1+buy_treshold)*p[t]), t_list[t]] for t in range(len(p))]

        #q_buy_fut_fut=[q_buy_fut(p,dt,Nmax,commission) for dt in range(1, n_fut+1)]
        #q_buy_fut_fut=np.array(q_buy_fut_fut)
        #q_buy_fut_pres=q_buy_fut(p,0,Nmax,commission)
        #q_buy=[[[q_buy_fut_pres[t,i],max(q_buy_fut_fut[:,t,i])] for i in range(Nmax)] for t in range(len(p))]
        #q_buy=np.array(q_buy)
        #q_buy_scalar=[sum((q_buy[t,i,0]-(q_buy[t,i,0]-q_buy[t,i,1])*(np.heaviside(-q_buy[t,i,1],1)-np.heaviside(q_buy[t,i,0]-q_buy[t,i,1],1)*np.heaviside(q_buy[t,i,1],1))-q_buy[t,i,1]*np.heaviside(q_buy[t,i,0],1)*np.heaviside(q_buy[t,i,1],1)*np.heaviside(q_buy[t,i,1]-q_buy[t,i,0],1))*gamma**(i+1) for i in range(Nmax)) for t in range(len(q_buy))]
        #result_buy=[[p[t],p_fut[t], q_buy_scalar[t]] for t in range(len(p))]


        #Save datasets
        buffer_folder=data_folder+coin+"/action_training/"
        save_file=buffer_folder+'sell_'+coin+"_"+timename+'.csv'
        with open(save_file, 'w', newline='') as f:
            wr = csv.writer(f)
            result=result_sell
            for t in range(len(result)):
                line=result[t]
                wr.writerow(line)
        save_file=buffer_folder+'buy_'+coin+"_"+timename+'.csv'
        with open(save_file, 'w', newline='') as f:
            wr = csv.writer(f)
            result=result_buy
            for t in range(len(result)):
                line=result[t]
                wr.writerow(line)

print("Finish!")

