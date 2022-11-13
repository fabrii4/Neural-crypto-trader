#import tensorflow as tf
import os
import sys
import re
import pandas as pd
import numpy as np
import json
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

#load lexicon extracted from train set and update vader lexicon
new_words = {}
folder='lexicons/'
lexicons=['negative.json', 'positive.json', 'neutral.json', 'new_words_crypto.json', 'lexicon_crypto_nospace.json']
for lex in lexicons:
    with open(folder+lex) as f:
        word_list=json.load(f)
    new_words.update(word_list)
vader.lexicon.update(new_words)

#load list of multi-words lexicon entries
multi_words_file="split_words.csv"
multi_words=pd.read_csv(folder+multi_words_file) 
multi_words=multi_words.to_numpy()[:,0]

multi_words_dict={}
multi_words_regex=[]
for multi_word in multi_words:
    unite_word = multi_word.replace(" ", "")
    regex=re.escape(multi_word)
    if re.search('^\w',regex):
        regex=r'\b'+regex
    if re.search('\w$',regex):
        regex=regex+r'\b'
    multi_words_regex.append(regex)
    multi_words_dict[multi_word]=unite_word
multi_words_pattern=re.compile("|".join(multi_words_regex))



#evaluate sentiment with vader method
data_folder="../../../data/"
base_folder=data_folder+"data_twitter/"
coin='btc/'
Dts={'1h':60*60, '4h':4*60*60}
Dt=Dts['1h'] #1h

if len(sys.argv)>1:
    coin=sys.argv[1]+"/"

def evaluate_sentiment(coin, timestep='1h'):
    print("Coin: "+coin)
    coin_folder=base_folder+coin+"/"
    result=[]
    load_path=base_folder+'sentiment_'+coin+'_'+timestep+'.csv'
    try: 
        result = pd.read_csv(load_path, sep='\t', index_col=0).values.tolist()
    except:
        print("Sentiment file not existing yet")
    
    tot_count=0
    res_count=0
    #find min max saved timestamps
    files = os.listdir(base_folder+coin)
    max_timestamp=0
    min_timestamp=1627477200
    for fil in files:
        timestamp=int(fil.split(coin+"_")[1].split(".csv")[0])
        max_timestamp = timestamp if timestamp > max_timestamp else max_timestamp
        min_timestamp = timestamp if timestamp < min_timestamp else min_timestamp

    timestamp=min_timestamp
    if len(result) > 0:
        last_timestamp=int(result[-1][0])
        timestamp = last_timestamp+Dt
    while(True):
        df_list=[]
        #load all files in range t+Dts[timestep] with step 1h
        for t in range(timestamp, timestamp+Dts[timestep], Dt):
            filename=coin+"_"+str(t)+".csv"
            if os.path.isfile(coin_folder+filename):
                df=pd.read_csv(coin_folder+filename, sep=',')
                df_list.append(df)
        if len(df_list) > 0:
            tot_count+=1
            res_count+=1
            #concat all loaded dataframe
            df=pd.concat(df_list)
            sentiment_average=0
            count=0
            for index, row in df.iterrows():
                if row['language'] != "en":
                    continue
                #preprocess input
                sentence=row['tweet'].lower()
                #regex is slower than direct string operation
                #sentence=multi_words_pattern.sub(lambda m: multi_words_dict[m.group(0)], sentence)
                #this is too slow
                for multi_word in multi_words:
                    if multi_word in sentence:
                        sentence=sentence.replace(multi_word, multi_words_dict[multi_word])
                #for black_word in blacklist:
                #    if black_word in sentence:
                #        sentence=sentence.replace(black_word, "")

                score=row['likes_count']#-score_min+1
                if score > 16:
                    score = 5
                elif score > 8:
                    score = 2.5
                else:
                    score = 1
                count+=score
                sentiment=vader.polarity_scores(sentence)
                sentiment=sentiment['compound']*score
                sentiment_average+=sentiment
            sentiment_average=sentiment_average/count if count > 0 else 0
            print(f"Step: {tot_count} | Timestamp: {timestamp} | Coin: {coin} | Sent: {sentiment_average:4.3f}\r", end='')
            number_post=len(df)
            score_average=count/number_post if number_post > 0 else 0
            data=[timestamp, sentiment_average, number_post, score_average]
            result.append(data)
        if res_count >=300:
            result_df = pd.DataFrame(result, columns = ['timestamp', 'mean sentiment', 
                                                        'number of posts', 'mean score'])
            #save to csv every 300 steps
            save_path=base_folder+'sentiment_'+coin+'_'+timestep+'.csv'
            result_df.to_csv(save_path, sep='\t', encoding='utf-8')
            print("Step: "+ str(tot_count)+" | Saving "+save_path+"                 ")
            res_count=0
        #step to next timestamp
        timestamp+=Dt
        if timestamp > max_timestamp:
            break

    #Final saving
    result_df = pd.DataFrame(result, columns = ['timestamp', 'mean sentiment', 
                                                'number of posts', 'mean score'])
    #save to csv
    save_path=base_folder+'sentiment_'+coin+'_'+timestep+'.csv'
    result_df.to_csv(save_path, sep='\t', encoding='utf-8')
    print("Finish!")



#evaluate sentiment for each coin
list_coin=['btc', 'eth', 'ltc', 'xrp', 'doge', 'ada', 'xlm', 'xmr', 'sol', 'xtz']
list_time=['1h', '4h']
for coin in list_coin:
    for time in list_time:
        evaluate_sentiment(coin, time)

