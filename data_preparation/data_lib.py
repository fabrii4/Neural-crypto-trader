from sklearn.preprocessing import MinMaxScaler
import os
import sys
from binance.client import Client
import csv
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import pathlib as pl
import yaml
import datetime as dt
from termcolor import colored
sys.path.insert(1, '../binance_interface/')
import util_lib as util
import sentiment_data_lib as sent

#interval deltas in millisec
Deltas= {
  "1m": 60000,
  "3m": 180000,
  "5m": 300000,
  "15m": 900000,
  "30m": 1800000,
  "1h": 3600000,
  "2h": 7200000,
  "4h": 14400000,
  "1d": 86400000
}

#bars labels
col_names=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

#feature indices
features_ind=[4,5,12,13,14,15,16,17,18,19,20]

data_folder="../data/"

#Data class to contain coin historical data
class Data:
    def __init__(self, credentials, coin, timestep='1h', length=256, pred_length=16, commission=0.01):
        self.coin = coin
        self.timestep = timestep
        self.length = length
        self.pred_length = pred_length
        self.cat_length = 5
        self.commission=commission
        self.credentials=credentials
        data, raw_data, raw_data_sent, stamp = get_prepare_data(credentials, coin, timestep, length, pred_length)
        self.raw_data = raw_data.copy() #shape (2*length, features) element zero is the oldest
        self.raw_data_sent = raw_data_sent.copy() #2*length, element zero is the oldest
        self.timestamp = int(stamp) #timestamp of last bars update (in ms)
        self.curr_timestamp = int(stamp) #timestamp of last price update (in ms)
        self.curr_price = data[0,-1]
        self.past_price = [data[0,-1]] #history of current prices
        self.data = data.copy() #shape (features, length) element zero is the oldest 
        norm_data, scaler_list = normalize(data)
        self.norm_data = norm_data.copy() #shape (1, length, features) el zero is most recent
        self.scaler = scaler_list.copy()
        self.prediction=np.zeros(pred_length) #shape (1,pred_length,1) el zero is most recent
        self.prediction_cat=np.zeros(self.cat_length) #(<-15%, <-5%, -5%<x<5%, >5%, >15%)
        self.coin_data=Data_coin(credentials, coin)
        self.wait_to_sell=0 #to check if a condition has been fulfilled before taking action 
        print(colored(coin+" data correctly initialized!", "green"))

    def update(self):
        start=self.timestamp
        timestamp_now=int(time.time()*1000)
        Delta=Deltas[self.timestep]
        old_curr_price=self.curr_price
        #connect to client
        try:
            client = Client(api_key=self.credentials[0], api_secret=self.credentials[1])
        except:
            print(colored("cannot connect (data update, cannot reach Client, retrying in 120s)", "red"))
            return False
        #update present data
        curr_timestamp=timestamp_now
        curr_price=self.curr_price
        try:
            curr_price = float(client.get_symbol_ticker(symbol=self.coin)['price'])
        except:
            print(colored("cannot connect (data update, cannot get current price, retrying in 120s)", "red"))
            return False
        coin_data=self.coin_data
        try:
            coin_data.update(client, self.coin)
        except:
            print(colored("cannot connect (data update, cannot get coin data, retrying in 120s)", "red"))
            return False
        self.curr_price = curr_price
        self.curr_timestamp=curr_timestamp
        self.coin_data=coin_data
        self.past_price.append(curr_price)
        self.past_price=self.past_price[-20:]
        #update historical data (only if current_time-closing_time>Delta)
        if timestamp_now - start >= 2*Delta:
            raw_bars=None
            try:
                raw_bars=update_data(client, self.coin, self.timestep, start+Delta,
                                     2*self.length, self.raw_data, timestamp_now)
            except:
                print(colored("cannot connect (data update, cannot get historical bars, retrying in 120s)", "red"))
                return False
            timestamp=int(raw_bars[-1,0])
            #if a new bar has been obtained update also sentiment data
            if timestamp > start:
                bars_sent=None
                raw_bars_sent=None
                try:
                    bars_sent, raw_bars_sent = sent.update_sentiment_data(self.coin,
                                        timestamp, 2*self.length, self.raw_data_sent)
                except:
                    print(colored("cannot connect (data update, cannot get sentiment data, retrying in 120s)", "red"))
                    return False
                #compare last timestamps from technical and sentiment data
                timestamp_sent=raw_bars_sent['timestamp'].iloc[-1]
                print(colored("Last timestamps tech|sent "+self.coin+": "+str(int(timestamp/1000))+"|"+str(timestamp_sent), "cyan"))
                #process technical data
                self.raw_data=raw_bars.copy()
                bars = prepare_tech_ind(raw_bars, self.length, self.pred_length)
                bars = filter_features(bars, features_ind, self.length)
                self.timestamp=timestamp
                #process sentiment data
                bars_sent=bars_sent[-self.length:]
                self.raw_data_sent=raw_bars_sent.copy()
                bars_sent=np.transpose(bars_sent)
                #combine data
                bars=np.concatenate((bars,bars_sent), axis=0)
                bars=bars.astype('float32')
                #update stored data
                self.data=bars.copy()
                norm_data, scaler_list = normalize(bars)
                self.norm_data = norm_data.copy()
                self.scaler = scaler_list.copy()

        if self.timestamp > start:
            print(colored("Coin "+self.coin+" updated!", "green"))
        return True if self.timestamp > start else False
        #return True if self.curr_price != old_curr_price else False

class Data_coin:
    def __init__(self, credentials, coin):
        #connect to client
        try:
            client = Client(api_key=credentials[0], api_secret=credentials[1])
        except:
            print(colored(f"cannot connect ({coin} data init, cannot contact client, exiting)", "red"))
            exit()
        try:
            balance, net_worth, coin_held, total_invested = get_account_data(client,coin)
        except:
            print(colored(f"cannot connect ({coin} data init, cannot get coin data, exiting)", "red"))
            exit()
        self.balance = balance
        self.net_worth = net_worth
        self.coin_held = coin_held
        self.total_invested = total_invested 

    def update(self, client, coin):
        balance, net_worth, coin_held, total_invested = get_account_data(client,coin)
        #self.balance = balance
        self.net_worth = net_worth
        #self.coin_held = coin_held
        #self.total_invested = total_invested 

#TECHNICAL DATA PROCESSING        

def get_all_data(client, coin, timestep, length):
    #timestamp in millisec rounded to minutes
    round_time=int(time.time()/100)*100
    timestamp_now=round_time*1000
    Delta=Deltas[timestep]
    start=timestamp_now-(length+1)*Delta
    bars = client.get_historical_klines(coin, timestep, start)
    #remove latest bar if closing time is in the future
    if int(bars[-1][6]) > timestamp_now:
        bars=bars[:-1]
    if len(bars) > length:
        bars = bars[-length:]
    return np.array(bars).astype('float64')

def update_data(client, coin, timestep, start, length, bars, timestamp_now):
    print(colored("Coin: "+coin+" Technical update" , "cyan"))
    new_bars = client.get_historical_klines(coin, timestep, start)
    #remove latest bar if closing time is in the future
    if int(new_bars[-1][6]) > timestamp_now:
        new_bars=new_bars[:-1]
    if len(new_bars)>0:
        new_bars=np.array(new_bars)
        bars = np.concatenate((bars, new_bars))
    if len(bars) > length:
        bars = bars[-length:]
    return bars.astype('float64')

def prepare_tech_ind(bars, length, pred_length):
    df = pd.DataFrame(bars, columns = col_names)
    df.ta.macd(append=True,timeperiod=pred_length) 
    df.ta.roc(append=True,timeperiod=pred_length) 
    df.ta.sma(append=True,timeperiod=pred_length) 
    df.ta.rsi(append=True,timeperiod=pred_length) 
    df.ta.cci(append=True,timeperiod=pred_length) 
    df.ta.ema(append=True,timeperiod=pred_length) 
    df.ta.atr(append=True,timeperiod=pred_length) 
    bars = np.nan_to_num(df)
    if len(bars) > length:
        bars = bars[-length:]
    return bars

def filter_features(bars, features_ind, length):
    bars = bars[:, features_ind]
    #if there are not enouth timesteps, expand bars history with zeros
    if bars.shape[0] < length:
        zero_fill=np.zeros((length-bars.shape[0], bars.shape[1]))
        bars=np.concatenate((zero_fill, bars), axis=0)
    bars=np.transpose(bars)
    return bars

#SENTIMENT DATA PROCESSING

def prepare_sentiment_ind(coin, timestamp_last, length):
    df_sent, raw_sent = sent.get_sentiment_data(coin, timestamp_last, length)
    return df_sent, raw_sent

#COMBINE AND NORMALIZE TECH-SENT DATA

def get_prepare_data(credentials, coin, timestep, length, pred_length):
    #connect to client
    try:
        client = Client(api_key=credentials[0], api_secret=credentials[1])
    except:
        print(colored("cannot connect (data init "+coin+", cannot contact client, exiting)", "red"))
        exit()
    #tech data
    try:
        bars=get_all_data(client, coin, timestep, 2*length)
    except:
        print(colored("cannot connect (data init "+coin+", cannot get historical bars, exiting)", "red"))
        exit()
    raw_bars = bars.copy()
    bars = prepare_tech_ind(bars, length, pred_length)
    timestamp=int(bars[-1,0])
    bars = filter_features(bars, features_ind, length)
    #sent data
    try:
        bars_sent, raw_bars_sent = prepare_sentiment_ind(coin, timestamp, 2*length)
    except:
        print(colored("cannot connect (data init "+coin+", cannot get sentiment data, exiting)", "red"))
        exit()
    bars_sent=bars_sent[-length:]
    bars_sent=np.transpose(bars_sent)
    #compare last timestamps from technical and sentiment data
    timestamp_sent=raw_bars_sent['timestamp'].iloc[-1]
    print(colored("Alignment check, timestamps tech|sent "+coin+": "+str(int(timestamp/1000))+"|"+str(timestamp_sent), "cyan"))
    #combine tech sent
    bars=np.concatenate((bars,bars_sent), axis=0)
    bars=bars.astype('float32')
    return bars, raw_bars, raw_bars_sent, timestamp

#return bars in flipped order: t0, t-1, t-2....
def normalize(bars):
    scaler_list=[]
    bars=np.reshape(bars,(len(bars),-1,1))
    for i in range(len(bars)):
        scaler = MinMaxScaler((0.33,0.66))
        bars[i]=np.flip(bars[i])
        bars[i]=scaler.fit_transform(bars[i])
        scaler_list.append(scaler)
    bars=np.reshape(bars,(len(bars),-1))
    bars=np.transpose(bars)
    bars=np.reshape(bars,(1,len(bars),-1))
    return bars, scaler_list

#returns bars in standard order: ...t-2, t-1, t0
def unnormalize(bars, scaler_list):
    bars=np.reshape(bars,(len(bars),-1,1))
    for i in range(len(bars)):
        scaler = scaler_list[i]
        bars[i]=scaler.inverse_transform(bars[i])
        bars[i]=np.flip(bars[i])
    bars=np.reshape(bars,(len(bars),-1))
    return bars


#GENERAL ACCOUNT INFO AND COIN CONFIGURATION

def get_account_data(client,coin):
    coin_info=client.get_symbol_info(coin)
    base=coin_info['baseAsset']
    quote=coin_info['quoteAsset']
    balance_info=client.get_asset_balance(asset=quote)
    coin_held_info=client.get_asset_balance(asset=base)
    balance=float(balance_info['free'])+float(balance_info['locked'])
    coin_held=float(coin_held_info['free'])+float(coin_held_info['locked'])
    curr_price = float(client.get_symbol_ticker(symbol=coin)['price'])

    #temporary testing solution (read coin info from stored config file)
    path=data_folder+coin+'/'
    path_config=path+'coin-config.yaml'
    try:
        config={}
        with open(path_config) as config_file:
            config = yaml.load(config_file)
        balance=config['balance']
        coin_held=config['coin_held']
        total_invested=config['total_invested']
    except:
        balance = 1000
        coin_held = 0
        total_invested = 0

    #balance = 1000
    net_worth = curr_price*coin_held
    #coin_held = 0
    #total_invested = 0
    return balance, net_worth, coin_held, total_invested



#get/set coin configuration file
def get_config(data):
    path=data_folder+data.coin+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    path_config=path+'coin-config.yaml'
    check_path = pl.Path(path_config)
    check_path.touch(exist_ok=True)
    with open(path_config) as config_file:
        config = yaml.load(config_file)
    if config == None:
        config = {}
    config['balance']=data.coin_data.balance
    config['coin_held']=data.coin_data.coin_held
    config['net_worth']=data.coin_data.net_worth
    config['total_invested']=data.coin_data.total_invested
    is_connected=True
    #connect to client
    try:
        client = Client(api_key=data.credentials[0], api_secret=data.credentials[1])
        #util.api_on(client)
    except:
        print(colored("cannot connect (config data, cannot contact client, not updating)", "red"))
        is_connected=False
    #get orders
    if util.internet_on() and is_connected:
        open_orders=[]
        all_orders=[]
        is_open_orders_update=True
        is_all_orders_update=True
        try:
            open_orders=client.get_open_orders(symbol=data.coin)
        except:
            print(colored("cannot connect (config data, cannot get open_orders, not updating)", "red"))
            is_open_orders_update=False
        try:
            all_orders=client.get_all_orders(symbol=data.coin)
        except:
            print(colored("cannot connect (config data, cannot get all orders, not updating)", "red"))
            is_all_orders_update=False
        #@TODO sort order lists by date
        #filled_orders=sorted(filled_orders, key=d.get, reverse=True)
        #update filled orders
        if is_all_orders_update:
            filled_orders=[order for order in all_orders if order['status'] == 'FILLED']
            formatted_orders=[]
            for order in filled_orders:
                formatted_order={'executedQty': float(order['executedQty']), 'origQty': float(order['origQty']), 'quoteQty': float(order['cummulativeQuoteQty']), 'price': float(order['price']), 'side': order['side'], 'time': str(dt.datetime.fromtimestamp(int(order['updateTime']/1000)))}
                formatted_orders.append(formatted_order)
            config['trade_history']=formatted_orders
        #update open orders
        if is_open_orders_update:
            formatted_open_orders=[]
            for order in open_orders:
                formatted_order={'origQty': float(order['origQty']), 'price': float(order['price']), 'side': order['side'], 'time': str(dt.datetime.fromtimestamp(int(order['time']/1000)))}
                formatted_open_orders.append(formatted_order)
            config['open_orders']=formatted_open_orders
    #save coin config file
    with open(path_config, 'w') as config_file:
        yaml.dump(config, config_file)
    return config
