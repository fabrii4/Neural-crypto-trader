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

def get_twitter_posts(coin, search, start_date, end_date, Dt=Dt):
    folder=base_folder+coin+"/"
    #define search parameters
    c = twint.Config()
    c.Search = search
    c.Since = start_date
    c.Until = end_date
    c.Lang = "en"
    c.Store_csv = True
    filename=coin+"_"+start_date+".csv"
    c.Output = folder+filename
    #run search
    twint.run.Search(c)
    time.sleep(1)

def check_twitter_posts(coin, search, Dt=Dt):
    print("--------------------------------------------------------------")
    print("Coin: "+coin)
    folder=base_folder+coin+"/"
    files = os.listdir(folder)
    max_timestamp=0
    for fil in files:
        timestamp=int(fil.split(coin+"_")[1].split(".csv")[0])
        max_timestamp = timestamp if timestamp > max_timestamp else max_timestamp
    timestamp_now=max_timestamp
    start_epoch=1502942400
    #if folder is not empty get last saved timestamp
    print("Start epoch: "+str(start_epoch))
    while(True):
        end_epoch=start_epoch+Dt
        if end_epoch >= timestamp_now:
            #print("Finish!")
            break
        start_date=str(start_epoch)
        end_date=str(end_epoch)
        #print(str(start_date)+"-"+str(end_date))
        filename=coin+"_"+str(start_epoch)+".csv"
        if not os.path.isfile(folder+filename):
            print("file "+filename+" missing!")
            try: 
                get_twitter_posts(coin, search, start_date, end_date)
            except:
                print("Exception!")
                end_epoch = start_epoch
        start_epoch = end_epoch 
    print("--------------------------------------------------------------")
    print("\n\n")


#get twitter post
get_twitter_posts("btc", "btc|bitcoin")
get_twitter_posts("eth", "eth|ethereum")
get_twitter_posts("ltc", "ltc|litecoin")
get_twitter_posts("xrp", "xrp|ripple")
get_twitter_posts("doge", "doge|dogecoin")
check_twitter_posts("ada", "ada|cardano")
check_twitter_posts("xmr", "xmr|monero")
check_twitter_posts("xlm", "xlm|stellar")

