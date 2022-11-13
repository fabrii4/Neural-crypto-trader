import os
import sys
import numpy as np
import csv
import requests
from termcolor import colored


#load sudo password
sudoFile = open('../sudo.txt', 'r')
sudo_password = sudoFile.readline().rstrip()
sudoFile.close()


#create ramdisk to share data with dash web interface
def init_ramdisk(path='/tmp/memory/'):
    if not os.path.exists(path):
        os.makedirs(path)
        sudoPassword = sudo_password
        command = 'mount -t tmpfs -o size=1G tmpfs '+path
        os.system('echo %s|sudo -S %s' % (sudoPassword, command))
        #os.system('sudo mount -t tmpfs -o size=1G tmpfs '+path)

def save_data(data, buy=None, sell=None, path='/tmp/memory/'):
    plot_length=100
    if not os.path.exists(path):
        init_ramdisk(path)
    path=path+data.coin+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    #raw_data, element zero is the oldest
    past = data.raw_data[-plot_length:]
    #prediction, element zero is the latest, need to be flipped
    future = np.flip(data.prediction,axis=0)
    future = np.insert(future,0, data.curr_price, axis=0)

    #keep track of old future data
    try:
        fut_file = data.coin+"_"+data.timestep+"_fut.csv"
        fut_path=path+fut_file
        future_old = np.genfromtxt(fut_path, delimiter=',')
        past_file = data.coin+"_"+data.timestep+"_past.csv"
        past_path=path+past_file
        past_old = np.genfromtxt(past_path, delimiter=',')
        timestamp_old=past_old[-1,0]
        Dt=past_old[-1,0]-past_old[-2,0]
        timestamp_now=past[-1,0]
        if len(future_old.shape)==1:
            future_old=np.reshape(future_old,(-1,1))
        #add a new old future column only if the data has been updated
        if timestamp_now>timestamp_old:
            while(True):
                #syncronize timestamps
                if timestamp_now-timestamp_old>Dt:
                    missing_data=np.zeros(future.shape)
                    future_old=np.concatenate((missing_data,future_old), axis=1)
                    timestamp_old+=Dt
                else:
                    break
            future=np.concatenate((future,future_old), axis=1)
            future=future[:,:plot_length]
        else:
            future_old[:,0]=future[:,0]
            future=future_old
    except:
        pass

    #resize buy/sell points
    start_epoch=past[0,0]
    if len(buy)>2:
        start_index=np.array(buy)[:,0].searchsorted(start_epoch, 'right')
        if start_index > 0:
            buy=buy[start_index:]
    elif len(buy)>0:
        if buy[0][0]<start_epoch:
            buy[0]=[-100,-100]
        if buy[-1][0]<start_epoch:
            buy[-1]=[-100,-100]
    if len(sell)>2:
        start_index=np.array(sell)[:,0].searchsorted(start_epoch, 'right')
        if start_index > 0:
            sell=sell[start_index:]
    elif len(sell)>0:
        if sell[0][0]<start_epoch:
            sell[0]=[-100,-100]
        if sell[-1][0]<start_epoch:
            sell[-1]=[-100,-100]


    #save past
    past_file = data.coin+"_"+data.timestep+"_past.csv"
    past_path=path+past_file
    with open(past_path, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(past)):
            line=past[i]
            wr.writerow(line)
    #save future
    fut_file = data.coin+"_"+data.timestep+"_fut.csv"
    fut_path=path+fut_file
    with open(fut_path, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(future)):
            line=future[i]
            wr.writerow(line)
    #save sent
    #raw_data_sent, element zero is the oldest
    sent = data.raw_data_sent.iloc[-plot_length:]
    sent_file = data.coin+"_"+data.timestep+"_sent.csv"
    sent_path=path+sent_file
    with open(sent_path, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(sent)):
            line=sent.iloc[i]
            wr.writerow(line)
    #save buy points
    #buy=np.reshape(buy,(len(buy),1))[-plot_length:]
    buy_file = data.coin+"_"+data.timestep+"_buy.csv"
    buy_path=path+buy_file
    with open(buy_path, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(buy)):
            line=buy[i]
            wr.writerow(line)
    #save sell points
    #sell=np.reshape(sell,(len(sell),1))[-plot_length:]
    sell_file = data.coin+"_"+data.timestep+"_sell.csv"
    sell_path=path+sell_file
    with open(sell_path, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(sell)):
            line=sell[i]
            wr.writerow(line)

def load_data(data, path='/tmp/memory/'):
    path=path+data.coin+"/"
    if not os.path.exists(path):
        buy=[[-100,-100]]
        sell=[[-100,-100]]
        return buy, sell

    #load buy points
    buy_file = data.coin+"_"+data.timestep+"_buy.csv"
    buy_path=path+buy_file
    try:
        buy = np.genfromtxt(buy_path, delimiter=',')
        buy=np.reshape(buy, (-1,2)).tolist()
        if len(buy)==0:
            buy=[[-100,-100]]
    except:
        buy=[[-100,-100]]
    #load sell points
    sell_file = data.coin+"_"+data.timestep+"_sell.csv"
    sell_path=path+sell_file
    try:
        sell = np.genfromtxt(sell_path, delimiter=',')
        sell=np.reshape(sell, (-1,2)).tolist()
        if len(sell)==0:
            sell=[[-100,-100]]
    except:
        sell=[[-100,-100]]
    return buy, sell
    
#check if internet is working 
def internet_on():
    try:
        requests.get('http://1.1.1.1', timeout=3)
        return True
    except (requests.ConnectionError, requests.Timeout) as exception: 
        print(colored('\nCannot connect, internet seems down', 'red'))
        return False

#check if api is working 
def api_on(client):
    try:
        status = client.get_system_status()
        return True
    except requests.exceptions.ConnectTimeout:
        print(colored('\nCannot contact api', 'red'))
        return False
