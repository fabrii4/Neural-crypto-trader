import csv
import sys
import os
from numpy import genfromtxt, reshape, zeros, insert, append, loadtxt, array, flip, where, savetxt, nan_to_num
import pandas as pd
import pandas_ta as ta

coins=['BTCEUR', 'BTCUSDT', 'LTCEUR', 'LTCUSDT', 'XRPEUR', 'XRPUSDT', 'ETHEUR', 'ETHUSDT', 'DOGEEUR', 'DOGEUSDT', 'ADAEUR', 'ADAUSDT', 'XLMEUR', 'XLMUSDT', 'XMRUSDT', 'SOLEUR', 'SOLUSDT', 'XTZUSDT']
#folders=['btc_eur', 'btc_usd', 'ltc_eur', 'ltc_usd', 'xrp_eur', 'xrp_usd', 'eth_eur', 'eth_usd']
folders=coins.copy()

for i in range(len(coins)):
    folder_input='../../data/'+folders[i]+'/'
    coin=coins[i]
    #times=['4h', '2h', '1h', '30m', '15m', '5m']
    times=['4h', '1h', '5m']

    folder_out=folder_input+'tech_ind/'
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    for time in times:
        # Load data
        file_path=folder_input+coin+'_'+time+'.csv'
        print(file_path)
        df = pd.read_csv(file_path, sep=",")
        label=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        df.columns = label
        #print(df.info)


        #set number of cores
        df.ta.cores = 4

        # Calculate Returns and append to the df DataFrame
        #df.ta.log_return(cumulative=True, append=True)
        #df.ta.percent_return(cumulative=True, append=True)

        # New Columns with results
        #df.columns

        # Take a peek
        #df.tail()
        window_size=16
        df.ta.macd(append=True,timeperiod=window_size) 
        df.ta.roc(append=True,timeperiod=window_size) 
        df.ta.sma(append=True,timeperiod=window_size) 
        df.ta.rsi(append=True,timeperiod=window_size) 
        df.ta.cci(append=True,timeperiod=window_size) 
        df.ta.ema(append=True,timeperiod=window_size) 
        df.ta.atr(append=True,timeperiod=window_size) 
        #df.ta.strategy("Momentum")
        #df.fillna(0, inplace=True)
        df = nan_to_num(df)
        df=df.clip(float('-1e200'),float('1e200'))
        file_path=folder_out+coin+'_'+time+'_tiout.csv'
        savetxt(file_path, df, delimiter=",")

print('Finish')


