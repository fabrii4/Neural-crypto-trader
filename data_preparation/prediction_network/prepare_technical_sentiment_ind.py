import csv
import sys
import os
import json
from numpy import genfromtxt, reshape, zeros, insert, append, loadtxt, array, flip, where, savetxt, nan_to_num
import pandas as pd
import pandas_ta as ta

coins=['BTCEUR', 'BTCUSDT', 'LTCEUR', 'LTCUSDT', 'XRPEUR', 'XRPUSDT', 'ETHEUR', 'ETHUSDT', 'DOGEEUR', 'DOGEUSDT', 'ADAEUR', 'ADAUSDT', 'XLMEUR', 'XLMUSDT', 'XMRUSDT', 'SOLEUR', 'SOLUSDT', 'XTZUSDT']
folders=coins.copy()
times=['4h', '1h', '5m']
times=['1h', '4h']

sentiments=['sentiment_btc', 'sentiment_btc', 'sentiment_ltc', 'sentiment_ltc', 'sentiment_xrp', 'sentiment_xrp', 'sentiment_eth', 'sentiment_eth', 'sentiment_doge', 'sentiment_doge', 'sentiment_ada', 'sentiment_ada', 'sentiment_xlm', 'sentiment_xlm', 'sentiment_xmr', 'sentiment_sol', 'sentiment_sol', 'sentiment_xtz']
sentiment_news=['sentiment_crypto_news', 'sentiment_finance_news']

data_folder='../../data/'

#Dt=60*60 #1h
Dts={'5m':5*60, '1h':60*60, '4h':4*60*60}

#val_ind: index of val in data starting from n_prev
#sma_ind: index of sma in data starting from n_prev
def sma(dt, data, val, n_prev, val_ind, sma_ind):
    sma=0
    if len(data) >= dt:
        sma=data[-1][n_prev+sma_ind]+(val-data[-dt][n_prev+val_ind])/dt
    elif len(data) == 0:
        sma=val
    else:
        sma=(sum(data[i][n_prev+val_ind] for i in range(len(data)))+val)/(len(data)+1)
    return sma


#iterate over coins
for i in range(len(coins)):
    #input tech ind
    folder_input=data_folder+folders[i]+'/tech_ind/'
    coin=coins[i]

    #input sentiments
    folder_sent=data_folder+'data_twitter/'
    folder_input_twitter=folder_sent
    folder_input_reddit=data_folder+'data_reddit/'
    #folder_input_lunar=data_folder+'data_lunar/'
    sentiment=sentiments[i]

    #output
    folder_out=data_folder+folders[i]+'/tech_sent/'
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    for time in times:
        Dt=Dts[time]
        # Load tech data
        file_path_tech=folder_input+coin+'_'+time+'_tiout.csv'
        print("Input tech: "+file_path_tech)
        df_tech = pd.read_csv(file_path_tech, sep=",")
        label_tech=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore', 'macd_0', 'macd_1', 'macd_2', 'roc', 'sma', 'rsi', 'cci', 'ema', 'atr']
        df_tech.columns = label_tech

        # Load sent data
        #reddit
        file_path_reddit=folder_input_reddit+sentiment+'_'+time+'.csv'
        print("Input reddit: "+file_path_reddit)
        df_reddit = pd.read_csv(file_path_reddit, sep="\t")
        df_reddit = df_reddit.sort_values('timestamp').reset_index(drop=True)
        #crypto_news
        file_path_reddit_cn=folder_input_reddit+sentiment_news[0]+'_'+time+'.csv'
        df_reddit_cn = pd.read_csv(file_path_reddit_cn, sep="\t")
        df_reddit_cn = df_reddit_cn.sort_values('timestamp').reset_index(drop=True)
        #financial_news
        file_path_reddit_fn=folder_input_reddit+sentiment_news[1]+'_'+time+'.csv'
        df_reddit_fn = pd.read_csv(file_path_reddit_fn, sep="\t")
        df_reddit_fn = df_reddit_fn.sort_values('timestamp').reset_index(drop=True)

        #twitter
        file_path_twitter=folder_input_twitter+sentiment+'_'+time+'.csv'
        print("Input twitter: "+file_path_twitter)
        df_twitter = pd.read_csv(file_path_twitter, sep="\t")
        df_twitter = df_twitter.sort_values('timestamp').reset_index(drop=True)

        #fear and greed
        file_path_fear_greed=folder_sent+'fear&greed_index.json'
        data = json.load(open(file_path_fear_greed))
        df_fear_greed = pd.DataFrame(data['data'])
        #df_fear_greed = pd.read_json(file_path_fear_greed)
        list_fear_greed=[]
        for i in range(len(df_fear_greed)):
            timestamp=int(df_fear_greed.iloc[i]['timestamp'])
            value=int(df_fear_greed.iloc[i]['value'])
            value_classification=df_fear_greed.iloc[i]['value_classification']
            list_fear_greed.append([timestamp, value, value_classification])
        list_fear_greed.reverse()
        df_fear_greed=pd.DataFrame(list_fear_greed)
        df_fear_greed.columns = ['timestamp', 'value', 'value_classification']

        #lunarcrush
        #file_path_lunar=folder_input_lunar+sentiment+'.csv'
        #print("Input sent: "+file_path_lunar)
        #df_lunar = pd.read_csv(file_path_lunar, sep=",")
        #df_lunar = df_lunar.sort_values('time').reset_index(drop=True)

        #combine datasets
        #start time
        min_time_tech=int(df_tech['Open time'].min()/1000)
        min_time_reddit=int(df_reddit['timestamp'].min())
        min_time_twitter=int(df_twitter['timestamp'].min())
        #min_time_lunar=int(df_lunar['time'].min())
        start_time=max([min_time_tech, min_time_reddit, min_time_twitter])
        #start_time=max([min_time_tech, min_time_twitter, min_time_lunar])
        #start_time=max([min_time_tech, min_time_twitter])
        #end time
        max_time_tech=int(df_tech['Open time'].max()/1000)
        max_time_reddit=int(df_reddit['timestamp'].max())
        max_time_twitter=int(df_twitter['timestamp'].max())
        #max_time_lunar=int(df_lunar['time'].max())
        end_time=min([max_time_tech, max_time_reddit, max_time_twitter])
        #end_time=min([max_time_tech, max_time_twitter, max_time_lunar])
        #end_time=min([max_time_tech, max_time_twitter])
        print("Start time: "+str(start_time))
        print("End time: "+str(end_time))

        start_time_fear_greed=1517443200

        #combine datasets lines by timestamp
        t=start_time
        data=[]
        indices=[0,0,0,0,0,0]
        #iterate over dataset timestamps
        while(True):
            #find tech data at time t, if missing average neighbour data
            if int(t*1000) in df_tech['Open time'].values:
                line_tech=df_tech.loc[df_tech['Open time'] == int(t*1000)]
                indices[0]=line_tech.index[0]
                line_tech=line_tech.iloc[0,:].to_list()
            else:
                line_prev=df_tech.loc[indices[0]].to_list()
                line_next=df_tech.loc[indices[0]+1].to_list()
                line_tech=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
                line_tech[0]=int(t*1000)

            #find reddit data at time t, if missing average neighbour data
            t_h=int(t/(60*60))*60*60
            if int(t_h) in df_reddit['timestamp'].values:
                line_reddit=df_reddit.loc[df_reddit['timestamp'] == int(t)]
                indices[1]=line_reddit.index[0]
                line_reddit=line_reddit.iloc[0,:].to_list()
            else:
                line_prev=df_reddit.loc[indices[1]].to_list()
                line_next=df_reddit.loc[indices[1]+1].to_list()
                line_reddit=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
                line_reddit[1]=int(t_h)

            #find reddit cypto news data at time t, if missing average neighbour data
            t_h=int(t/(60*60))*60*60
            if int(t_h) in df_reddit_cn['timestamp'].values:
                line_reddit_cn=df_reddit_cn.loc[df_reddit_cn['timestamp'] == int(t)]
                indices[2]=line_reddit_cn.index[0]
                line_reddit_cn=line_reddit_cn.iloc[0,:].to_list()
            else:
                line_prev=df_reddit_cn.loc[indices[2]].to_list()
                line_next=df_reddit_cn.loc[indices[2]+1].to_list()
                line_reddit_cn=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
                line_reddit_cn[1]=int(t_h)

            #find reddit finance news data at time t, if missing average neighbour data
            t_h=int(t/(60*60))*60*60
            if int(t_h) in df_reddit_fn['timestamp'].values:
                line_reddit_fn=df_reddit_fn.loc[df_reddit_fn['timestamp'] == int(t)]
                indices[3]=line_reddit_fn.index[0]
                line_reddit_fn=line_reddit_fn.iloc[0,:].to_list()
            else:
                line_prev=df_reddit_fn.loc[indices[3]].to_list()
                line_next=df_reddit_fn.loc[indices[3]+1].to_list()
                line_reddit_fn=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
                line_reddit_fn[1]=int(t_h)

            #find lunar data at time t, if missing average neighbour data
            #sentiment time t_h steps by 1h
#            t_h=int(t/(60*60))*60*60
#            if int(t_h) in df_lunar['time'].values:
#                line_lunar=df_lunar.loc[df_lunar['time'] == int(t_h)]
#                indices[5]=line_lunar.index[0]
#                line_lunar=line_lunar.iloc[0,:].to_list()
#            else:
#                line_prev=df_lunar.loc[indices[5]].to_list()
#                line_next=df_lunar.loc[indices[5]+1].to_list()
#                line_lunar=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
#                line_lunar[1]=int(t_h)
#            #remove unnecessary data
#            #data_indices=[10,11,12,13,14,21,22,23,24,25,26,27,28,29,30,34,36,37,38,39,40]
#            data_indices=[14,21,22,23,24,25,40]
#            line_lunar=[line_lunar[i] for i in data_indices]

            #find twitter data at time t, if missing average neighbour data
            #sentiment time t_h steps by 1h
            t_h=int(t/(60*60))*60*60
            if int(t_h) in df_twitter['timestamp'].values:
                line_twitter=df_twitter.loc[df_twitter['timestamp'] == int(t_h)]
                indices[4]=line_twitter.index[0]
                line_twitter=line_twitter.iloc[0,:].to_list()
            else:
                line_prev=df_twitter.loc[indices[4]].to_list()
                line_next=df_twitter.loc[indices[4]+1].to_list()
                line_twitter=[(g + h) / 2 for g, h in zip(line_prev, line_next)]
                line_twitter[1]=int(t_h)
            
            #find fear_greed index at itme t
            line_fear_greed=[50]
            if t >= start_time_fear_greed:
                line_fear_greed=df_fear_greed.loc[df_fear_greed['timestamp'] <= int(t)].iloc[-1,:].to_list()
                line_fear_greed=line_fear_greed[1:-1]

            #compute sma tech
            n_prev=0
            sma100=sma(100, data, line_tech[4], n_prev, 4, 21)
            sma16=sma(16, data, line_tech[4], n_prev, 4, 22)
            line_tech.append(sma100)
            line_tech.append(sma16)

            #compute sma twitter
            line_twitter=line_twitter[2:-1]
            n_prev=len(line_tech)
            sma100=sma(100, data, line_twitter[0], n_prev, 0, 2)
            sma16=sma(16, data, line_twitter[0], n_prev, 0, 3)
            line_twitter.append(sma100)
            line_twitter.append(sma16)

            #compute sma reddit
            line_reddit=line_reddit[2:-1]+line_reddit_cn[2:-1]+line_reddit_fn[2:-1]
            n_prev=len(line_tech+line_twitter+line_fear_greed)
            sma100=sma(100, data, line_reddit[0], n_prev, 0, 6)
            sma16=sma(16, data, line_reddit[0], n_prev, 0, 7)
            line_reddit.append(sma100)
            line_reddit.append(sma16)
            #compute sma crypto news reddit
            sma100=sma(100, data, line_reddit[2], n_prev, 2, 8)
            sma16=sma(16, data, line_reddit[2], n_prev, 2, 9)
            line_reddit.append(sma100)
            line_reddit.append(sma16)
            #compute sma finance news reddit
            sma100=sma(100, data, line_reddit[4], n_prev, 4, 10)
            sma16=sma(16, data, line_reddit[4], n_prev, 4, 11)
            line_reddit.append(sma100)
            line_reddit.append(sma16)

#            #compute sma lunar
#            sent=[0, 0.25, 0.5, 0.75, 1]
#            sent_lunar=line_lunar[1:-1]
#            sent_mean=0.5
#            if line_lunar[0]>0:
#                sent_mean=sum(val[0] * val[1] for val in zip(sent, sent_lunar))/line_lunar[0]
#            #line_lunar: n_tweet, sent_mean, galaxy_score
#            line_lunar=[line_lunar[0],sent_mean,line_lunar[-1]]
#            n_prev=len(line_tech+line_twitter+line_fear_greed)
#            #sma n_tweet
#            sma100=sma(100, data, line_lunar[0], n_prev, 0, 3)
#            sma16=sma(16, data, line_lunar[0], n_prev, 0, 4)
#            line_lunar.append(sma100)
#            line_lunar.append(sma16)
#            #sma sent_mean
#            sma100=sma(100, data, line_lunar[1], n_prev, 1, 5)
#            sma16=sma(16, data, line_lunar[1], n_prev, 1, 6)
#            line_lunar.append(sma100)
#            line_lunar.append(sma16)
#            #sma galaxy_score
#            sma100=sma(100, data, line_lunar[2], n_prev, 2, 7)
#            sma16=sma(16, data, line_lunar[2], n_prev, 2, 8)
#            line_lunar.append(sma100)
#            line_lunar.append(sma16)
            
            #combine data at time t
            line=line_tech + line_twitter + line_fear_greed + line_reddit
            #line=line_tech + line_twitter + line_fear_greed + line_lunar
            #line=line_tech + line_twitter + line_fear_greed
            data.append(line)

            t+=Dt
            if t > end_time:
                break

        #create output dataset
        label_tech_new=['close sma100', 'close sma16']
        label_twitter_fear=['twitter sentiment', 'twitter posts', 'twitter sentiment sma100', 'twitter sentiment sma16', 'fear greed index']
        label_reddit=['reddit sentiment', 'reddit posts', 'cypto sentiment', 'crypto posts','finance sentiment', 'finance posts', 'reddit sentiment sma100', 'reddit sentiment sma16', 'crypto sentiment sma100', 'crypto sentiment sma16', 'finance sentiment sma100', 'finance sentiment sma16']
        #label_lunar=['reddit_posts', 'reddit_posts_score', 'reddit_comments', 'reddit_comments_score', 'tweets', 'tweet_sentiment1', 'tweet_sentiment2', 'tweet_sentiment3', 'tweet_sentiment4', 'tweet_sentiment5', 'tweet_sentiment_impact1', 'tweet_sentiment_impact2', 'tweet_sentiment_impact3', 'tweet_sentiment_impact4', 'tweet_sentiment_impact5', 'sentiment_relative', 'news', 'price_score', 'social_impact_score', 'correlation_rank', 'galaxy_score']
        #label_lunar=['tweets', 'mean sentiment', 'galaxy score', 'tweets sma100', 'tweets sma16', 'mean sentiment sma100', 'mean sentiment sma16', 'galaxy score sma100', 'galaxy score sma16']
        #label=label+label_twitter_fear+label_lunar
        label=label_tech+label_tech_new+label_twitter_fear+label_reddit
        #label=label+label_twitter_fear
        df_out=pd.DataFrame(data)
        df_out.columns=label
        #save output
        file_path_out=folder_out+coin+'_'+time+'_tsi.csv'
        print("Output: "+file_path_out)
        #savetxt(file_path_out, df_out, delimiter=",")
        df_out.to_csv(file_path_out, sep=',', encoding='utf-8', index=False)

print('Finish')
