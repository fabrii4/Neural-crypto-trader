import os
import sys
from binance.client import Client
#from binance.websockets import BinanceSocketManager
import csv
import time
import datetime as dt
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import numpy as np
import time as tm
import yaml
from termcolor import colored
from multiprocessing import Process
sys.path.insert(1, '../data_preparation/')
sys.path.insert(1, '../prediction_network/')
sys.path.insert(1, '../action_network/')
import data_lib as dl
import tcn_net_lib as tn
import action_net_lib as an
import util_lib as util


#load api keys
keyFile = open('../binance_keys.txt', 'r')
api_key = keyFile.readline().rstrip()
api_secret = keyFile.readline().rstrip()
keyFile.close()

#@TODO move to util_lib
def print_info(coin_config, data):
    active_trading=coin_config['active_trading'] if 'active_trading' in coin_config else False
    buy_price=data.coin_data.total_invested/data.coin_data.coin_held if data.coin_data.coin_held>0 else 0
    print(colored(data.coin+"-"+data.timestep, "yellow")+
          colored("| Trading: ","cyan")+
          colored(str(active_trading), "green" if active_trading else "red")+
          colored("| Price: ","cyan")+
          colored(str(data.curr_price),"yellow")+
          colored("| Buy: ","cyan")+
          colored(str(buy_price) if buy_price>0 else "N/A","green" if buy_price < data.curr_price and buy_price > 0 else "red" if buy_price > data.curr_price else "yellow")+
          colored("| Balance: ","cyan")+
          colored(f"{data.coin_data.balance+data.coin_data.total_invested:.2f}","yellow"))


#trading process
def trade_coin(coin='XRPEUR', timestep='1h'):
    #initialize binance client
    #client = Client(api_key=api_key, api_secret=api_secret)
    credentials=(api_key,api_secret)
    #initialize coin data
    data = dl.Data(credentials, coin, timestep)
    #initialize prediction model
    pred_model = tn.load_model(data)
    #initialize action model
    buy_model, sell_model = an.load_model(data)

    #used to plot buy/sell actions
    buy, sell = util.load_data(data)
    is_update=True
    action=0
    print(colored("Running...", "cyan"))
    while(True):

        #read coin configuration file
        coin_config=dl.get_config(data)

        is_exec=False
        if is_update:
            #predict future trend
            data.prediction, data.prediction_cat = tn.predict(pred_model, data)
            #predict action action (if active_trading is active)
            if 'active_trading' in coin_config and coin_config['active_trading']:
                action = an.action(buy_model, sell_model, data)

        #take action
        data, is_exec = an.take_action(data, action)
        
        #add action points to plot 
        buy, sell = an.update_buy_sell_points(data, is_exec, action, buy, sell)
        if is_exec:
            action=0
        
        #save data used for plot
        util.save_data(data, buy, sell)

        #print status info
        print_info(coin_config, data)

        #update data
        tm.sleep(120)
        if util.internet_on():# and util.api_on(client):
            is_update=data.update()

#update all available pairs list
def update_pair_list():
    while(True):
        print(colored("Updating available pairs list", "cyan"))
        try:
            client = Client(api_key=api_key, api_secret=api_secret)
        except:
            print(colored("cannot connect (pairs list update, retrying in 24h)", "red"))
            tm.sleep(86400) #sleep 1m
            continue
        if util.internet_on() and util.api_on(client):
            is_update_success=True
            try:
                pairs_info=client.get_exchange_info()
            except:
                is_update_success=False
            if is_update_success:
                pair_list=[]
                for pair_info in pairs_info['symbols']:
                    pair = pair_info['symbol']
                    if pair_info['status'] == 'TRADING':
                        pair_list.append(pair)
                with open('config.yaml') as config_file:
                    config = yaml.load(config_file)
                config['list_pairs_all']=pair_list
                with open('config.yaml', 'w') as config_file:
                    yaml.dump(config, config_file)
        tm.sleep(86400) #sleep 24h

#update account info
def update_account_info():
    while(True):
        try:
            client = Client(api_key=api_key, api_secret=api_secret)
        except:
            print(colored("cannot connect (update account info, retrying in 120s)", "red"))
            tm.sleep(120) #sleep 1m
            continue
        if util.internet_on() and util.api_on(client):
            trading_assets=[] #assets which hold coin
            #calculate total current account balance
            try:
                info = client.get_account()
            except:
                print(colored("cannot connect (update account info, retrying in 120s)", "red"))
                tm.sleep(120) #sleep 1m
                continue
            balances=info['balances']
            total_usdt_value=0
            is_update_success=True
            for balance in balances:
                bal_free=float(balance['free'])
                bal_lock=float(balance['locked'])
                if bal_free > 0 or bal_lock > 0:
                    asset=balance['asset']
                    trading_assets.append(asset)
                    bal=bal_free+bal_lock
                    usdt_price=1
                    if asset != 'USDT':
                        usdt_pair= asset+'USDT'
                        usdt_price=1.0
                        try:
                            usdt_price=client.get_symbol_ticker(symbol=usdt_pair)['price']
                            usdt_price=float(usdt_price)
                        except:
                            print(colored("cannot connect (update account info, retrying in 120s)", "red"))
                            is_update_success=False
                            continue
                        usdt_value=bal*usdt_price
                        total_usdt_value+=usdt_value
                    if asset == 'USDT':
                        usdt_available=bal
                        total_usdt_value+=bal
                    if asset == 'EUR':
                        eur_available=bal
            eur_usdt_price=1.0
            try:
                eur_usdt_price=float(client.get_symbol_ticker(symbol='EURUSDT')['price'])
            except:
                print(colored("cannot connect (update account info, retrying in 120s)", "red"))
                is_update_success=False
            total_eur_value=total_usdt_value/eur_usdt_price
            #update config file
            try:
                with open('config.yaml') as config_file:
                    config = yaml.load(config_file)
            except:
                is_update_success=False
            if is_update_success:
                config['total_usdt']=total_usdt_value
                config['total_eur']=total_eur_value
                config['available_usdt']=usdt_available
                config['available_eur']=eur_available
                config['assets_with_balance']=trading_assets
                with open('config.yaml', 'w') as config_file:
                    yaml.dump(config, config_file)
        tm.sleep(120) #sleep 1m

#process launchers
def launch_trade_process(coin='XRPEUR', timestep='1h'):
    p = Process(target=trade_coin, args=(coin, timestep))
    p.start()
    print(colored("launched process for coin "+coin+"-"+timestep+" pid "+str(p.pid), "green"))
    return p

def launch_update_process():
    p = Process(target=update_pair_list)
    p.start()
    print(colored("launched process for update coin list "+"pid "+str(p.pid), "green"))
    return p

def launch_account_update_process():
    p = Process(target=update_account_info)
    p.start()
    print(colored("launched process for update account info "+"pid "+str(p.pid), "green"))
    return p


#MAIN
if __name__ == '__main__':

    #initialize dict of running trading processes
    dict_trade_process={ }
    #initialize lists of trading pairs
    list_pairs = []
    list_times = []

    #launch update process to update list of all available trading pairs
    #@TODO re-enable this
    #launch_update_process()
    #launch update process for general account info
    account_process=launch_account_update_process()
    
    while(True):

        #read list_pair from conf
        #@TODO consider using lockfile to safely read/write config files
        try:
            with open('config.yaml') as config_file:
                config = yaml.load(config_file)
                if 'trading_pairs' in config and 'trading_times' in config:
                    list_pairs = config['trading_pairs']
                    list_times = config['trading_times']
                    list_pair=[(pair, time) for pair in list_pairs for time in list_times]
        except:
            pass

        #Launch trade processes if not already running and add it to dict_trade_process
        for pair in list_pair:
            if not pair in dict_trade_process:
                process = launch_trade_process(pair[0], pair[1])
                dict_trade_process[pair] = process

        time.sleep(10)

        #terminate process if does not exist in list_pair
        for pair in dict_trade_process:
            process = dict_trade_process[pair]
            if not pair in list_pair:
                is_terminate=True
                timeout=tm.time()+10
                while(process.is_alive()):
                    process.terminate()
                    tm.sleep(0.1)
                    if tm.time() > timeout:
                        print(colored("unable to terminate process "+pair[0]+"-"+pair[1], "red"))
                        is_terminate=False
                        break
                if is_terminate:
                    print(colored("terminated process "+pair[0]+"-"+pair[1], "yellow"))

        #remove dead processes from dict_trade_process
        new_dict_trade_process={}
        for pair in dict_trade_process:
            process = dict_trade_process[pair]
            if process.is_alive():
                new_dict_trade_process[pair]=process
        dict_trade_process = new_dict_trade_process

        #check if account process is running and restart it otherwise
        if not account_process.is_alive():
            print(colored("account process died, restarting", "yellow"))
            account_process=launch_account_update_process()




