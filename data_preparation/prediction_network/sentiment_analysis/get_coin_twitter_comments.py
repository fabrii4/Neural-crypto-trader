import twint
import datetime as dt
import pandas as pd
import time
import os


coin="btc"
search="btc|bitcoin"
data_folder="../../../data/"
base_folder=data_folder+"data_twitter/"
Dt=60*60 #1h

def get_twitter_posts(coin, search, Dt=Dt):
    print("Coin: "+coin)
    folder=base_folder+coin+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp_now=int(time.time())
    start_epoch=1502942400
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
        if end_epoch >= timestamp_now:
            print("Finish!")
            break
        start_date=str(start_epoch)
        end_date=str(end_epoch)
        print("Start epoch: "+start_date+" | "+str(dt.datetime.fromtimestamp(start_epoch)))
        print("End epoch: "+end_date+" | "+str(dt.datetime.fromtimestamp(end_epoch)))
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
        try:
            twint.run.Search(c)
            time.sleep(1)
        except:
            end_epoch = start_epoch
        #step to next epoch
        start_epoch = end_epoch


#get twitter post
get_twitter_posts("btc", "btc|bitcoin")
print("------------------------------------------------------------")
get_twitter_posts("eth", "eth|ethereum")
print("------------------------------------------------------------")
get_twitter_posts("ltc", "ltc|litecoin")
print("------------------------------------------------------------")
get_twitter_posts("xrp", "xrp|ripple")
print("------------------------------------------------------------")
get_twitter_posts("doge", "doge|dogecoin")
print("------------------------------------------------------------")
get_twitter_posts("xmr", "xmr|monero")
print("------------------------------------------------------------")
get_twitter_posts("xlm", "xlm|stellar")
print("------------------------------------------------------------")
get_twitter_posts("ada", "ada|cardano")
print("------------------------------------------------------------")
get_twitter_posts("sol", "sol|solana")
print("------------------------------------------------------------")
get_twitter_posts("xtz", "xtz|tezos")
