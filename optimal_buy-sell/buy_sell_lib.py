import os
import csv
import numpy as np
import math
#from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager

coin='XRPEUR'

def buy(data, Dt):
    price_list=data.past_price
    time_list=[data.curr_timestamp-i*Dt for i in range(len(price_list))]
    buy_price=data.data[0,-1]
    min_price=min(price_list)
    sma=curr_price
    sigma=0
    n_sma=16

    price_list=price_list[-n_sma:]
    time_list=time_list[-n_sma:]
    #moving average
    sma=sum(price_list)/len(price_list)
    sma_t=sum(time_list)/len(time_list)
    #moving standard deviation
    sigma=math.sqrt(sum((price_list[i]-sma)**2 for i in range(len(price_list)))/len(price_list))
    #moving regression line slope
    beta=sum((time_list[i]-sma_t)(price_list[i]-sma) for i in range(len(price_list)))/sum((time_list[i]-sma_t)**2 for i in range(len(price_list)))
    #buy condition
    if beta>0 and price_list[-1] > min_price+sigma:
        return True

    return False

def sell(data, Dt):
    price_list=data.past_price
    time_list=[data.curr_timestamp-i*Dt for i in range(len(price_list))]
    buy_price=data.data[0,-1]
    max_price=max(price_list)
    sma=curr_price
    sigma=0
    n_sma=16

    price_list=price_list[-n_sma:]
    time_list=time_list[-n_sma:]
    #moving average
    sma=sum(price_list)/len(price_list)
    sma_t=sum(time_list)/len(time_list)
    #moving standard deviation
    sigma=math.sqrt(sum((price_list[i]-sma)**2 for i in range(len(price_list)))/len(price_list))
    #moving regression line slope
    beta=sum((time_list[i]-sma_t)(price_list[i]-sma) for i in range(len(price_list)))/sum((time_list[i]-sma_t)**2 for i in range(len(price_list)))

    #sell condition
    if beta<0 and price_list[-1] > buy_price*(1+data.commission):
        return True

    return False



#def buy_socket(coin, curr_timestamp, curr_price, Dt):
#    price_list=[curr_price]
#    time_list=[curr_timestamp]
#    min_price=curr_price
#    sma=curr_price
#    sigma=0
#    n_sma=16
#    step=int(Dt/6) #20 sec
#    buy_step=step
#    #initialize websocket
#    websocket = BinanceWebSocketApiManager(exchange="binance.com")
#    websocket.create_stream(['trade'], [coin], output="UnicornFy")
#    while True:
#        #get data from websocket
#        stream_data = websocket.pop_stream_data_from_stream_buffer()
#        if stream_data:
#            try:
#                #get and store data
#                coin_price=float(stream_data['price'])
#                time=int(stream_data['event_time'])
#                price_list.append(coin_price)
#                time_list.append(time)
#                price_list=price_list[-n_sma:]
#                time_list=time_list[-n_sma:]
#                #min price value
#                if min(price_list) < min_price:
#                    min_price = min(price_list)
#                #moving average
#                sma=sum(price_list)/len(price_list)
#                sma_t=sum(time_list)/len(time_list)
#                #moving standard deviation
#                sigma=math.sqrt(sum((price_list[i]-sma)**2 for i in range(len(price_list)))/len(price_list))
#                #moving regression line slope
#                beta=sum((time_list[i]-sma_t)(price_list[i]-sma) for i in range(len(price_list)))/sum((time_list[i]-sma_t)**2 for i in range(len(price_list)))

#                #buy condition
#                if time >= buy_step:
#                    buy_step+=step
#                    if beta>0 and price_list[-1] > min_price+sigma:
#                        return True, coin_price
#                if time-curr_timestamp>Dt:
#                    break
#                #print(event)
#            except:
#                #print("error")
#                pass

#    return False, curr_price

#def sell_socket(coin, curr_timestamp, curr_price, buy_price, Dt):
#    price_list=[curr_price]
#    time_list=[curr_timestamp]
#    max_price=curr_price
#    sma=curr_price
#    sigma=0
#    n_sma=16
#    step=int(Dt/6) #20 sec
#    sell_step=step
#    #initialize websocket
#    websocket = BinanceWebSocketApiManager(exchange="binance.com")
#    websocket.create_stream(['trade'], [coin], output="UnicornFy")
#    while True:
#        #get data from websocket
#        stream_data = websocket.pop_stream_data_from_stream_buffer()
#        if stream_data:
#            try:
#                #get and store data
#                coin_price=float(stream_data['price'])
#                time=int(stream_data['event_time'])
#                price_list.append(coin_price)
#                time_list.append(time)
#                price_list=price_list[-n_sma:]
#                time_list=time_list[-n_sma:]
#                #min price value
#                if max(price_list) > max_price:
#                    max_price = max(price_list)
#                #moving average
#                sma=sum(price_list)/len(price_list)
#                sma_t=sum(time_list)/len(time_list)
#                #moving standard deviation
#                sigma=math.sqrt(sum((price_list[i]-sma)**2 for i in range(len(price_list)))/len(price_list))
#                #moving regression line slope
#                beta=sum((time_list[i]-sma_t)(price_list[i]-sma) for i in range(len(price_list)))/sum((time_list[i]-sma_t)**2 for i in range(len(price_list)))

#                #sell condition
#                if time >= sell_step:
#                    sell_step+=step
#                    if beta<0 and coin_price > buy_price:
#                        return True, coin_price
#                if time-curr_timestamp>Dt:
#                    break
#                #print(event)
#            except:
#                #print("error")
#                pass

#    return False, curr_price
