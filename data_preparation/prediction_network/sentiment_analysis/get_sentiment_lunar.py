import pandas as pd
import time
import os
import sys
import re
import math
import numpy as np
import json
import csv
import requests

data_folder="../../../data/"
base_folder=data_folder+"data_lunar/"

def get_lunar_sentiment(coin, timestamp_start, Dt=3600*23):
    start=str(timestamp_start)
    end=str(timestamp_start+Dt)
    if timestamp_start+Dt > int(time.time()):
        end=str((int(time.time()/3600)-1)*3600)
    url_request="https://api.lunarcrush.com/v2?data=assets&key=rcvzsths0juwnn3dsf3ml&symbol="+coin+"&start="+start+"&end="+end
    response = requests.get(url_request)
    data = json.loads(response.text)
    df=pd.DataFrame(data['data'])
    df=pd.DataFrame(df['timeSeries'].iloc[0])
    return df

def get_lunar_sentiment_all(coin):
    timestamp_now=int(time.time())
    timestamp=1546329600
    Dt=3600 #1h
    print("Coin: "+coin)
    #initialize dataframe
    df_tot=get_lunar_sentiment(coin, timestamp, Dt*23)
    timestamp+=Dt*24
    #get latest saved timestamp
    file_path=base_folder+'sentiment_'+coin.lower()+'.csv'
    if os.path.isfile(file_path) and os.stat(file_path).st_size > 1:
        df_tot=pd.read_csv(file_path, sep=',', index_col=0)
        timestamp=df_tot['time'].iloc[-1]+Dt
    #get new data
    it=0
    while(True):
        if timestamp+Dt>timestamp_now:
            break
        #get max 24 hours per call
        df=get_lunar_sentiment(coin, timestamp, Dt*23)
        timestamp=df['time'].iloc[-1]+Dt if 'time' in df else timestamp+Dt*24
        df_tot=pd.concat([df_tot,df])
        df.fillna(0, inplace=True)
        print(f"Timestamp: {timestamp} | Data: {len(df_tot)}\r", end='')
        it+=1
        #save to csv periodically
        if it>100:
            df_tot.to_csv(file_path, sep=',', encoding='utf-8')
            it=0
    #save to csv final
    df_tot.to_csv(file_path, sep=',', encoding='utf-8')


#get all sentiment data
coin_list=['BTC', 'LTC', 'ETH', 'XRP', 'DOGE', 'ADA', 'XLM', 'XMR', 'SOL', 'XTZ', 'BNB', 'LIT']
for coin in coin_list:
    get_lunar_sentiment_all(coin)

print("Finish!")

