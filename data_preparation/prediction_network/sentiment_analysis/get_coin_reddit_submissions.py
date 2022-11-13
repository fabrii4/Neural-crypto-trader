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
from multiprocessing import Process
from termcolor import colored

data_folder="../../../data/"
base_folder=data_folder+"data_reddit/"

#coin subreddit and search pattern
coin_search={
    'CRYPTO_NEWS': ['cryptocurrencynews,CryptoCurrency,CryptoCurrencies', 'news|report|update', 'cryptocurrency news|crypto news'],
    'FINANCE_NEWS': ['Finance,FinanceNews', 'news|report|update', 'finance news|financial news|economic news'],
    'BTC': ['bitcoin,btc,CryptoCurrency', 'btc|bitcoin'],
    'LTC': ['litecoin,altcoin,CryptoCurrency', 'ltc|litecoin'],
    'ETH': ['ethereum,altcoin,CryptoCurrency', 'eth|ethereum'],
    'XRP': ['Ripple,altcoin,CryptoCurrency', 'xrp|ripple'],
    'DOGE': ['dogecoin,altcoin,CryptoCurrency', 'doge|dogecoin'],
    'ADA': ['cardano,altcoin,CryptoCurrency', 'ada|cardano'],
    'XLM': ['stellar,altcoin,CryptoCurrency', 'xlm|stellar'],
    'XMR': ['monero,altcoin,CryptoCurrency', 'xmr|monero'],
    'SOL': ['solana,altcoin,CryptoCurrency', 'sol|solana'],
    'XTZ': ['tezos,altcoin,CryptoCurrency', 'ztz|tezos'],
}

coin_list=['CRYPTO_NEWS', 'FINANCE_NEWS', 'BTC', 'LTC', 'ETH', 'XRP', 'DOGE', 'ADA', 'XLM', 'XMR', 'SOL', 'XTZ']


#grab hourly data from reddit
def get_reddit_sentiment(timestamp_start, subreddit, search, Dt=3600):
    start=str(timestamp_start)
    end=str(timestamp_start+Dt)
    #search parameters
    api_address="https://api.pushshift.io/reddit/submission/search?"
    subreddit_search="subreddit="+subreddit
    q_search="q="+search
    start_time="after="+start
    end_time="before="+end
    size="size=500"
    has_text="selftext:not=[removed]"
    out_fields="fields=author,subreddit,title,selftext,created_utc,score"
    sep="&"
    url_request=api_address + subreddit_search + sep + q_search + sep + start_time + sep+end_time + sep + size + sep + has_text + sep + out_fields
    if subreddit == "":
        url_request=api_address + q_search + sep + start_time + sep+end_time + sep + size + sep + has_text + sep + out_fields
    df=pd.DataFrame([])
    while(True):
        try:
            response = requests.get(url_request)
            data = json.loads(response.text)
            df=pd.DataFrame(data['data'])
        except:
            pass
        else:
            break
    return df

#get all data
def get_reddit_sentiment_all(coin):
    timestamp_now=int(time.time())
    timestamp=1502942400
    Dt=3600 #1h
    #set saving folder
    folder=base_folder+coin+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    #get last saved timestamp
    if os.listdir(folder):
        max_timestamp=timestamp
        files = os.listdir(folder)
        for fil in files:
            timestamp=int(fil.split(coin+"_")[1].split(".csv")[0])
            if timestamp > max_timestamp:
                max_timestamp = timestamp
        timestamp=max_timestamp+Dt
    #get search parameters
    search_list=coin_search[coin]
    subreddit=search_list[0]
    search=search_list[1]
    search1= search if len(search_list)<3 else search_list[2]
    print("\nCoin: "+coin)
    print("SubReddits: "+subreddit+ " Search: "+search)
    #get new data
    while(True):
        if timestamp+Dt>timestamp_now:
            break
        #get max 24 hours per call
        df=get_reddit_sentiment(timestamp, subreddit, search, Dt)
        if len(df)<30:
            df_expand=get_reddit_sentiment(timestamp, "", search1, Dt)
            if len(df)>0:
                df=pd.concat([df,df_expand])
            elif len(df_expand)>0:
                df=df_expand
        print(f"Coin: {coin} | Timestamp: {timestamp} | Data: {len(df)}               \r", end='')
        #save to csv final
        filename=coin+"_"+str(timestamp)+".csv"
        file_path=folder+filename
        if len(df)>0:
            df.to_csv(file_path, sep=',', encoding='utf-8')
        timestamp+=Dt


##process launchers
#def launch_process(coin):
#    p = Process(target=get_reddit_sentiment_all, args=(coin))
#    p.start()
#    print(colored("launched process for coin "+coin+" pid "+str(p.pid), "green"))
#    return p

#get all sentiment data
if len(sys.argv)>1:
    coin=sys.argv[1]
    get_reddit_sentiment_all(coin)
else:
    for coin in coin_list:
        get_reddit_sentiment_all(coin)

print("Finish!")

