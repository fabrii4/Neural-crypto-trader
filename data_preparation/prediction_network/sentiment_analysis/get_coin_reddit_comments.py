from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
import time
import os
import sys

api = PushshiftAPI()


coin="btc"
if len(sys.argv)>1:
    coin=sys.argv[1]

folder="./data_reddit/"+coin+"/"
Dt=60*60 #1h

search_words='btc|bitcoin'
if len(sys.argv)>2:
    search_words=sys.argv[2]

timestamp_now=int(time.time())
start_epoch=1502942400

#if folder is not empty
if os.listdir(folder):
    files = os.listdir(folder)
    paths = [os.path.join(folder, basename) for basename in files]
    latest_file = max(paths, key=os.path.getctime)
    timestamp=latest_file.split(coin+"_")[1].split(".csv")[0]

    start_epoch=int(timestamp)+Dt

while(True):

    end_epoch=start_epoch+Dt
    if end_epoch >= timestamp_now:
        print("Finish!")
        break

    gen=api.search_comments(q=search_words, 
                            after=start_epoch, before=end_epoch,
                            limit=500)
    results = list(gen)
    data_list=[]
    for res in results:
        res_d=res.d_
        body=res_d['body'][:500]
        body='"'+body.replace("\t"," ",).replace('"', "'")+'"'
        timestamp=int(res_d['created_utc'])
        datetime=dt.datetime.fromtimestamp(timestamp)
        score=res_d['score']
        subreddit=res_d['subreddit']
        data=[body, score, subreddit, datetime, timestamp]
        data_list.append(data)

    print(str(start_epoch) + " " + str(len(results)))

    df = pd.DataFrame(data_list, columns = ['body', 'score', 'subreddit', 'datetime', 'timestamp'])
    df=df.sort_values('score', ascending=False)
    filename=coin+"_"+str(start_epoch)+".csv"
    df.to_csv(folder+filename, sep='\t', encoding='utf-8')

    start_epoch = end_epoch 


