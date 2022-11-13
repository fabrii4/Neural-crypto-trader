import twint
import datetime as dt
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
import nltk
from termcolor import colored
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


coin="btc"
search="btc|bitcoin"
base_folder="../data/data_twitter/"
lexicon_folder='../data_preparation/prediction_network/sentiment_analysis/lexicons/'
Dt=60*60 #1h
coin_dict={'BTCUSDT': ['btc', 'btc|bitcoin'], 'BTCEUR': ['btc', 'btc|bitcoin'], 'LTCUSDT': ['ltc', 'ltc|litecoin'], 'LTCEUR': ['ltc', 'ltc|litecoin'], 'ETHUSDT': ['eth', 'eth|ethereum'], 'ETHEUR': ['eth', 'eth|ethereum'], 'XRPUSDT': ['xrp', 'xrp|ripple'], 'XRPEUR': ['xrp', 'xrp|ripple'], 'DOGEUSDT': ['doge', 'doge|dogecoin'], 'DOGEEUR': ['doge', 'doge|dogecoin'] }


def get_twitter_posts(timestamp_last, length, coin, search, Dt=Dt):
    print(colored("Coin: "+coin+" Sentiment update" , "cyan"))
    folder=base_folder+coin+"/"
    start_epoch=timestamp_last-Dt*length
    #if folder is not empty get last saved timestamp
    if os.listdir(folder):
        max_timestamp=start_epoch
        files = os.listdir(folder)
        for fil in files:
            timestamp=int(fil.split(coin+"_")[1].split(".csv")[0])
            if timestamp > max_timestamp:
                max_timestamp = timestamp
        start_epoch=max_timestamp+Dt
    while(True):
        end_epoch=start_epoch+Dt
        if end_epoch > timestamp_last:
            break
        start_date=str(start_epoch)
        end_date=str(end_epoch)
        #define search parameters
        c = twint.Config()
        c.Search = search
        c.Since = start_date
        c.Until = end_date
        c.Lang = "en"
        c.Store_csv = True
        c.Hide_output = True
        c.Count=True
        c.Stats=True
        filename=coin+"_"+str(start_epoch)+".csv"
        c.Output = folder+filename
        #run search
        twint.run.Search(c)
        #time.sleep(2)
        start_epoch = end_epoch


def get_fear_greed_index(length):
    n_days=math.ceil(length/24)+1
    url_request="https://api.alternative.me/fng/?limit="+str(n_days)+"&?format=json"
    #df_fear_greed = pd.read_json(url_request)
    response = requests.get(url_request)
    data = json.loads(response.text)
    df_fear_greed = data['data']
    list_fear_greed=[]
    for i in range(len(df_fear_greed)):
        timestamp=int(df_fear_greed[i]['timestamp'])
        value=int(df_fear_greed[i]['value'])
        value_classification=df_fear_greed[i]['value_classification']
        list_fear_greed.append([timestamp, value, value_classification])
    list_fear_greed.reverse()
    df_fear_greed=pd.DataFrame(list_fear_greed)
    df_fear_greed.columns = ['timestamp', 'value', 'value_classification']
    return df_fear_greed


#load lexicon extracted from train set and update vader lexicon
def update_lexicon(vader):
    new_words = {}
    lexicons=['negative.json', 'positive.json', 'neutral.json', 'new_words_crypto.json', 'lexicon_crypto_nospace.json']
    for lex in lexicons:
        with open(lexicon_folder+lex) as f:
            word_list=json.load(f)
        new_words.update(word_list)
    vader.lexicon.update(new_words)

#load list of multi-words lexicon entries
def create_multi_word_dict():
    multi_words_file="split_words.csv"
    multi_words=pd.read_csv(lexicon_folder+multi_words_file) 
    multi_words=multi_words.to_numpy()[:,0]

    multi_words_dict={}
    #multi_words_regex=[]
    for multi_word in multi_words:
        unite_word = multi_word.replace(" ", "")
        #regex=re.escape(multi_word)
        #if re.search('^\w',regex):
        #    regex=r'\b'+regex
        #if re.search('\w$',regex):
        #    regex=regex+r'\b'
        #multi_words_regex.append(regex)
        multi_words_dict[multi_word]=unite_word
    #multi_words_pattern=re.compile("|".join(multi_words_regex))
    return multi_words, multi_words_dict


def evaluate_sentiment(coin, start_timestamp, end_timestamp, Dt=Dt):
    vader = SentimentIntensityAnalyzer()
    update_lexicon(vader)
    multi_words, multi_words_dict = create_multi_word_dict()
    coin_folder=base_folder+coin+"/"
    result=[]
    timestamp=start_timestamp
    while(True):
        filename=coin+"_"+str(timestamp)+".csv"
        if os.path.isfile(coin_folder+filename):
            df=pd.read_csv(coin_folder+filename, sep=',')
            sentiment_average=0
            count=0
            for index, row in df.iterrows():
                if row['language'] != "en":
                    continue
                #preprocess input
                sentence=row['tweet'].lower()
                #this is slow
                for multi_word in multi_words:
                    if multi_word in sentence:
                        sentence=sentence.replace(multi_word, multi_words_dict[multi_word])
                score=row['likes_count']#-score_min+1
                score = 5 if score > 16 else 2.5 if score > 8 else 1
                count+=score
                #get sentiment polarity
                sentiment=vader.polarity_scores(sentence)
                sentiment=sentiment['compound']*score
                sentiment_average+=sentiment
            sentiment_average=sentiment_average/count if count > 0 else 0
            number_post=len(df)
            score_average=count/number_post if number_post > 0 else 0
            data=[int(timestamp), sentiment_average, number_post, score_average]
            result.append(data)
        #step to next timestamp
        timestamp+=Dt
        if timestamp >= end_timestamp:
            break
    #Final result
    result_df = pd.DataFrame(result, columns = ['timestamp', 'mean sentiment', 
                                                'number of posts', 'mean score'])

    return result_df



def sma(dt, data, val, n_tech, val_ind, sma_ind):
    sma=0
    if len(data) >= dt:
        sma=data[-1][n_tech+sma_ind]+(val-data[-dt][n_tech+val_ind])/dt
    elif len(data) == 0:
        sma=val
    else:
        sma=(sum(data[i][n_tech+val_ind] for i in range(len(data)))+val)/(len(data)+1)
    return sma


def compute_sma_sent(start_time, end_time, df_twitter, df_fear_greed, data=[], Dt=Dt):
    t=int(start_time)
    index=0
    line_twitter=None
    df_twitter = df_twitter.sort_values('timestamp').reset_index(drop=True)
    df_fear_greed = df_fear_greed.sort_values('timestamp').reset_index(drop=True)
    while(True):
        #find twitter data at time t, if missing average neighbour data
        if t in df_twitter['timestamp'].values:
            line_twitter=df_twitter.loc[df_twitter['timestamp'] == t]
            index=line_twitter.index[0]
            line_twitter=line_twitter.iloc[0,:].to_list()
        else:
            line_prev=df_twitter.iloc[index].to_list()
            line_next=df_twitter.iloc[index+1].to_list() if index+1 in df_twitter.index.values else None
            if line_prev[0] < t and line_next != None:
                line_twitter=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
                line_twitter[0]=t
            else:
                line_twitter=line_prev.copy()
                line_twitter[0]=t
        
        #find fear_greed index at time t
        start_time_fear_greed=df_fear_greed.iloc[0]['timestamp']
        line_fear_greed=[50]
        if t >= start_time_fear_greed:
            line_fear_greed = df_fear_greed.loc[df_fear_greed['timestamp'] <= t].iloc[-1,:].to_list()
            line_fear_greed=line_fear_greed[1:-1]

        #compute sma twitter
        line_twitter=line_twitter[1:-1]
        sma100=sma(100, data, line_twitter[0], 0, 0, 2)
        sma16=sma(16, data, line_twitter[0], 0, 0, 3)
        line_twitter.append(sma100)
        line_twitter.append(sma16)
        #combine data at time t

        #line=line_tech + line_reddit + line_twitter + line_fear_greed
        line=line_twitter + line_fear_greed
        data.append(line)

        t+=Dt
        if t >= end_time:
            break

    #create output dataset
    #label=['twitter sentiment', 'twitter posts', 'twitter sentiment sma100', 'twitter sentiment sma16', 'fear greed index']
    #df_out=pd.DataFrame(data)
    #df_out.columns=label
    df_out=np.array(data,dtype = object)

    return df_out



def get_sentiment_data(coin_pair, timestamp_last, length):
    #convert timestamp in millisec to timestamp in sec
    timestamp_last=int(timestamp_last/1000)+Dt
    coin=coin_dict[coin_pair][0]
    search=coin_dict[coin_pair][1]
    get_twitter_posts(timestamp_last, length, coin, search)
    start_timestamp=timestamp_last-length*Dt
    end_timestamp=timestamp_last
    df_twitter=evaluate_sentiment(coin, start_timestamp, end_timestamp)
    df_fear_greed=get_fear_greed_index(length)
    df_out = compute_sma_sent(start_timestamp, end_timestamp, df_twitter, df_fear_greed)

    return df_out, df_twitter

def update_sentiment_data(coin_pair, timestamp_last, length, df_twitter):
    #convert timestamp in millisec to timestamp in sec
    timestamp_last=int(timestamp_last/1000)+Dt
    coin=coin_dict[coin_pair][0]
    search=coin_dict[coin_pair][1]
    get_twitter_posts(timestamp_last, length, coin, search)
    start_timestamp=df_twitter['timestamp'].iloc[-1]+Dt
    end_timestamp=timestamp_last
    df_twitter_new=evaluate_sentiment(coin, start_timestamp, end_timestamp)
    df_twitter=pd.concat([df_twitter, df_twitter_new], ignore_index=True)
    df_twitter=df_twitter.iloc[-length:]
    df_fear_greed=get_fear_greed_index(length)
    df_out = compute_sma_sent(start_timestamp, end_timestamp, df_twitter, df_fear_greed)
    df_out=df_out[-length:]

    return df_out, df_twitter
