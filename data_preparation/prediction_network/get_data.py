import os
from binance.client import Client
#from binance.websockets import BinanceSocketManager
#from twisted.internet import reactor
import csv
import time
from numpy import genfromtxt

#load api keys
keyFile = open('../binance_keys.txt', 'r')
api_key = keyFile.readline().rstrip()
api_secret = keyFile.readline().rstrip()
keyFile.close()

client = Client(api_key, api_secret)

coins=['BTCEUR', 'BTCUSDT', 'LTCEUR', 'LTCUSDT', 'XRPEUR', 'XRPUSDT', 'ETHEUR', 'ETHUSDT', 'DOGEEUR', 'DOGEUSDT', 'ADAEUR', 'ADAUSDT', 'XLMEUR', 'XLMUSDT', 'XMRUSDT', 'SOLEUR', 'SOLUSDT', 'XTZUSDT']
#folders=['btc_eur', 'btc_usd', 'ltc_eur', 'ltc_usd', 'xrp_eur', 'xrp_usd', 'eth_eur', 'eth_usd']
folders=coins.copy()

list_time=['4h', '2h', '1h', '30m', '15m', '5m']#, '3m', '1m']
list_time_min=[240, 120, 60, 30, 15, 5]#, 3, 1]

for n in range(len(coins)):
    coin=coins[n]
    folder='../../data/'+folders[n]+'/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # request historical candle (or klines) data
    timestamp_now = time.time()*1000
    for i in range(len(list_time)):
        max_candle=1000
        millisec=1000
        sec=60
        minutes=list_time_min[i]
        min1000=max_candle*sec*millisec*minutes
        Delta=0
        #check if file already exists
        filename=folder+coin+'_'+list_time[i]+'.csv'
        if os.path.isfile(filename) and os.stat(filename).st_size > 0:
            dataset = genfromtxt(filename, delimiter=',')
            timestamp = int(dataset[-1][0] + list_time_min[i]*sec*millisec)
        else:
            timestamp = client._get_earliest_valid_timestamp(coin, list_time[i])
        print("file: "+filename)
        print("initial timestamp: "+str(timestamp))
        with open(filename, 'a+', newline='') as f:
            wr = csv.writer(f)
            while(True):
                print(Delta)
                start=timestamp + Delta*min1000
                end=timestamp + (Delta + 1)*min1000
                if start > timestamp_now:
                    break
                bars = client.get_historical_klines(coin, list_time[i], start, end)
                #remove latest bar if closing time is in the future
                if int(bars[-1][6]) > timestamp_now:
                    bars=bars[:-1]
                #save result
                for line in bars:
                    wr.writerow(line)
                Delta+=1

print("Finish")
