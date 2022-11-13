import os
import dash 
import dash_auth
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import html
import dash_daq as daq
from dash import dash_table as dt
import plotly 
import random 
import plotly.graph_objs as go 
from collections import deque 
import pandas as pd
import numpy as np
from datetime import datetime
import yaml


# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'user': 'password'
}

#bars labels
col_names=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

mem_folder='/tmp/memory/'

#get coin plot data 
def load_data(coin='XRPEUR', timestep='5m'):
    # Load data
    past_path=mem_folder+coin+'/'+coin+'_'+timestep+'_past.csv'
    fut_path=mem_folder+coin+'/'+coin+'_'+timestep+'_fut.csv'  
    buy_path=mem_folder+coin+'/'+coin+'_'+timestep+'_buy.csv'  
    sell_path=mem_folder+coin+'/'+coin+'_'+timestep+'_sell.csv'    
    past = pd.read_csv(past_path, sep=",", header=None)
    past.columns = col_names
    fut = np.genfromtxt(fut_path, delimiter=',')
    try:
        buy = pd.read_csv(buy_path, sep=",", header=None)
    except:
        buy=pd.DataFrame([[-100,-100]])
    buy.columns = ['time', 'price']
    buy=buy[buy['time'] > 0]
    try:
        sell = pd.read_csv(sell_path, sep=",", header=None)
    except:
        sell=pd.DataFrame([[-100,-100]])
    sell.columns = ['time', 'price']
    sell=sell[sell['time'] > 0]
    return past, fut, buy, sell

def read_write_general_config(new_pairs=[], new_times=[], path='../binance_interface/config.yaml'):
    #read list_pair from conf
    with open(path) as config_file:
        config = yaml.safe_load(config_file)
    pairs = config['trading_pairs']
    times = config['trading_times']
    pairs_all = config['list_pairs_all']
    times_all = config['list_times_all']
    if (new_pairs is not None and new_times is not None and 
        len(new_pairs) > 0 and len(new_times) > 0):
        if set(new_pairs) != set(pairs) or set(new_times) != set(times):
            config['trading_pairs']=new_pairs
            config['trading_times']=new_times
            pairs=new_pairs
            times=new_times
            with open(path, 'w') as config_file:
                yaml.dump(config, config_file)
    return pairs, times, pairs_all, times_all

def read_account_info_general_config(path='../binance_interface/config.yaml'):
    #read list_pair from conf
    with open(path) as config_file:
        config = yaml.safe_load(config_file)
    total_usdt_value=config['total_usdt']
    total_eur_value=config['total_eur']
    usdt_available=config['available_usdt']
    eur_available=config['available_eur']
    trading_assets=config['assets_with_balance']
    return total_usdt_value, total_eur_value, usdt_available, eur_available, trading_assets

def read_coin_config(coin):
    path='../data/'+coin+'/coin-config.yaml'
    bal=0
    coin_held=0
    net_worth=0
    tot_invested=0
    open_ord=[]
    ord_hist=[]
    active_trading=False
    if not os.path.exists(path):
        return bal, coin_held, net_worth, tot_invested, open_ord, ord_hist, active_trading
    with open(path) as config_file:
        config = yaml.safe_load(config_file)
    if config == None:
        return bal, coin_held, net_worth, tot_invested, open_ord, ord_hist, active_trading
    bal=config['balance']
    coin_held=config['coin_held']
    net_worth=config['net_worth']
    tot_invested=config['total_invested']
    open_ord=config['open_orders']
    ord_hist=config['trade_history']
    if 'active_trading' in config :
        active_trading=config['active_trading']
    return bal, coin_held, net_worth, tot_invested, open_ord, ord_hist, active_trading

def set_active_trading_config(coin, active_trading):
    path='../data/'+coin+'/coin-config.yaml'
    if not os.path.exists(path):
        return
    with open(path) as config_file:
        config = yaml.safe_load(config_file)
    if config == None:
        return
    if 'active_trading' not in config or config['active_trading'] != active_trading:
        config['active_trading']=active_trading
        with open(path, 'w') as config_file:
            yaml.dump(config, config_file)

#get initial configuration
pairs, times, pairs_all, times_all = read_write_general_config()
pairs_times=[pair+"-"+time for pair in pairs for time in times]

##################################
#start dash interface  
app = dash.Dash(__name__, update_title=None) 

#use user password authentication
#auth = dash_auth.BasicAuth(
#    app,
#    VALID_USERNAME_PASSWORD_PAIRS
#)

colors = {
    'background': '#111111',
    'background-R': '#212f3c',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.Div(className='row',  # Define the row element
        children=[

#########################
#left column
html.Div(className='four columns user-controls',
children = [
    html.H2('CRYPTO TRADER'),
    html.P("Neural trading of crypto currencies", className='sub-title'),
    html.Div(className='left-container', children=[
    html.P("Active trading pairs:"),
    html.Div(
    [
        dcc.Dropdown(className='div-for-dropdown single-choice',
            id='coin-choice',
            options=[{'label':name, 'value':name} for name in pairs_times],
            value=pairs_times[0],
            placeholder='Select active coin...',
            clearable=False,
            persistence = True,
            searchable=False
        )
    ]),
    html.P("Add/remove trading pairs:"),
    html.Div(
    [
        dcc.Dropdown(className='div-for-dropdown',
            id='coin-add',
            options=[{'label':name, 'value':name} for name in pairs_all],
            value=pairs,
            placeholder='Add/remove coins...',
            clearable=True,
            multi=True,
            persistence = True
        ),
        dcc.Dropdown(className='div-for-dropdown',
            id='time-add',
            options=[{'label':name, 'value':name} for name in times_all],
            value=times,
            placeholder='Add/remove timestep...',
            clearable=True,
            multi=True,
            persistence = True
        )
    ]),
    html.H6("Account info:"),
    html.Div(className='info-box', id='account-info-box')
        ])
]),  # Define the left element

###################
#right column
html.Div(className='eight columns chart bg-grey', # Define the right element
    children = [  
        dcc.Store(id = 'n-predictions'),
        dcc.Store(id = 'step-predictions'),
        dcc.Store(id = 'length-predictions'),
        dcc.Store(id = 'show-candles'),
        dcc.Graph(
            id = 'live-graph', 
            animate = False,
            config={'displayModeBar': False}
        ), 
        dcc.Interval( 
            id = 'graph-update', 
            interval = 10*1000, #update interval in milliseconds
            n_intervals = 0
        ), 
        dcc.Tabs(className='custom-tabs-container', parent_className='custom-tabs', 
                 id='tabs-coin', value='tab-1', style={'width':'100%'}, children=[
            dcc.Tab(className='custom-tab', selected_className='custom-tab--selected',      
                    label='Status', value='tab-1'),
            dcc.Tab(className='custom-tab', selected_className='custom-tab--selected', 
                    label='Open-orders', value='tab-2'),
            dcc.Tab(className='custom-tab', selected_className='custom-tab--selected', 
                    label='Trade-history', value='tab-3'),
            dcc.Tab(className='custom-tab', selected_className='custom-tab--selected', 
                    label='Settings', value='tab-4'),
        ]),
        html.Div(id='tabs-content')
])
])])

#################################
#callbacks 

#update account info content
@app.callback(Output('account-info-box', 'children'),
              Input('graph-update', 'n_intervals'))
def render_content(n):
    total_usdt, total_eur, usdt_av, eur_av, assets = read_account_info_general_config()
    assets_string=", ".join(assets)
    return html.Div(children = [
        html.P(f"Current total value: {total_usdt:10.2f}$ | {total_eur:10.2f}€"),
        html.P(f"Available Balance: {usdt_av:10.2f}$ | {eur_av:10.2f}€"),
        html.P("Invested Balance: -"),
        html.P("Initial Balance: -"),
        html.P("Current net worth: -"),
        html.P(f"Assets with balance: {assets_string}")
    ])


#update tab content
@app.callback(Output('tabs-content', 'children'),
              #[Input('graph-update', 'n_intervals'), 
              [Input('tabs-coin', 'value'), 
               Input('coin-choice', 'value')])
#def render_content(n, tab, coin_time):
def render_content(tab, coin_time):
    coin=coin_time.split('-')[0] if coin_time is not None else ""
    bal, coin_held, net_worth, tot_invested, open_ord, ord_hist, active_trading = read_coin_config(coin)
    if tab == 'tab-1':
        return html.Div(className='user-controls',
            children=[html.Div(children=[
            html.P(f'Balance: {bal:10.2f}'),
            html.P(f'Coin held: {coin_held:10.2f}'),
            html.P(f'Total worth: {net_worth:10.2f}'),
            html.P(f'Total invested: {tot_invested:10.2f}')],
            style={'display': 'inline-block', 'width': '33%'}),
            html.Div(children=[
            html.P('Limit order: '),
            html.Div(dcc.Input(id='price-box', type='numeric',
                     className='input-box', placeholder='Price...', value='')),
            html.Div(dcc.Input(id='coin-box', type='numeric',
                     className='input-box', placeholder='Coin amount...', value='')),
            html.Div(dcc.Input(id='fiat-box', type='numeric',
                     className='input-box', placeholder='Fiat amount...', value='')),
            html.Button('Buy', id='button-buy', className='button'),
            html.Button('Sell', id='button-sell', className='button')],
            style={'display': 'inline-block', 'width': '33%'}),
            html.Div(children=[
            daq.BooleanSwitch(className='switch', id='neural-switch', color='#1166f9',
            label='Neural Trading:', on = active_trading, labelPosition='top'),
            html.P(' '),
            html.P('(Activate/Deactivate automatic trading driven by neural networks)')],
            style={'display': 'inline-block', 'width': '33%', 'text-align': 'center'}
            )
        ])
    elif tab == 'tab-2':
        return html.Div(className='table-container', children=[
            dt.DataTable(id='open-order-table',
            columns=[
            {'name': 'origQty', 'id': 'origQty'},
            {'name': 'price', 'id': 'price'},
            {'name': 'side', 'id': 'side'},
            {'name': 'time', 'id': 'time'}
            ],
            data=open_ord,
            style_as_list_view=True,
            style_data={'border': 'none'},
            style_cell={'backgroundColor':'transparent',
                        'textAlign': 'left'},
            style_header={'backgroundColor':'transparent', 'fontWeight': 'bold', 
                          'textAlign': 'left', 'border': 'none'},
            style_data_conditional=[{
                'if': { 'state': 'selected'},  # 'active' | 'selected'},
                'backgroundColor': 'transparent', 'border': 'none'}]
            )
        ])
    elif tab == 'tab-3':
        return html.Div(className='table-container', children=[
            dt.DataTable(id='history-order-table',
            columns=[
            {'name': 'executedQty', 'id': 'executedQty'},
            {'name': 'origQty', 'id': 'origQty'},
            {'name': 'price', 'id': 'price'},
            {'name': 'quoteQty', 'id': 'quoteQty'},
            {'name': 'side', 'id': 'side'},
            {'name': 'time', 'id': 'time'}
            ],
            data=ord_hist, 
            style_as_list_view=True,
            style_data={'border': 'none'},
            style_cell={'backgroundColor':'transparent',
                        'textAlign': 'left'},
            style_header={'backgroundColor':'transparent', 'fontWeight': 'bold', 
                          'textAlign': 'left', 'border': 'none'},
            style_data_conditional=[{
                'if': { 'state': 'selected'},  # 'active' | 'selected'},
                'backgroundColor': 'transparent', 'border': 'none'}]
            )
        ])
    elif tab == 'tab-4':
        return html.Div(className='user-controls',
            children=[
            html.Div(children=[
            html.P('Number of historic predictions:', style={'font-weight': 'bold'}),
            dcc.Slider(min=0,max=100,value=0, id='set-n-predictions',
                marks={0: {'label': '0', 'style': {'color': '#77b0b1'}},
                    100: {'label': '100', 'style': {'color': '#f50'}}}),
            html.P(' '),
            html.P('Step between historic predictions:', style={'font-weight': 'bold'}),
            dcc.Slider(min=1,max=16,value=1, id='set-step-predictions',
                marks={1: {'label': '1', 'style': {'color': '#77b0b1'}},
                    16: {'label': '16', 'style': {'color': '#f50'}}})],
            style={'display': 'inline-block', 'width': '33%', 'text-align': 'center'}),
            html.Div(children=[
            html.P(' '),
            html.P('Prediction length:', style={'font-weight': 'bold'}),
            dcc.Slider(min=1,max=16,value=16, id='set-length-predictions',
                marks={1: {'label': '1', 'style': {'color': '#77b0b1'}},
                    16: {'label': '16', 'style': {'color': '#f50'}}}),
            html.P(' '),
            html.P('Show Candlesticks:', style={'font-weight': 'bold'}),
            daq.BooleanSwitch(className='switch', id='set-show-candles', color='#1166f9',
                              on = False, labelPosition='top')],
            style={'display': 'inline-block', 'width': '33%', 'text-align': 'center'}),
        ])

#update active trading switch config
@app.callback(Output('neural-switch', 'on'),
        Input('neural-switch', 'on'), State('coin-choice', 'value'))
def update_trading_switch(active_trading, coin_time):
    coin=coin_time.split('-')[0] if coin_time is not None else ""
    set_active_trading_config(coin, active_trading)
    return active_trading

#update number of predictions to display and step between them in dcc.Stores
@app.callback([Output('n-predictions', 'data'), Output('step-predictions', 'data'),
               Output('length-predictions', 'data'), Output('show-candles', 'data')],
        [Input('set-n-predictions', 'value'), Input('set-step-predictions', 'value'),
         Input('set-length-predictions', 'value'), Input('set-show-candles', 'on')])
def update_displayed_prediction_histories(set_n_pred, set_step_pred, set_len_pred, set_show_candles):
    return set_n_pred, set_step_pred, set_len_pred, set_show_candles

#update number of predictions to display and step between them in tab sliders
@app.callback([Output('set-n-predictions', 'value'), Output('set-step-predictions', 'value'), 
               Output('set-length-predictions', 'value'), Output('set-show-candles', 'on')],
        [Input('n-predictions', 'data'), Input('step-predictions', 'data'), 
         Input('length-predictions', 'data'), Input('show-candles', 'data')])
def update_sliders_prediction_histories(n_pred, step_pred, len_pred, show_candles):
    return n_pred, step_pred, len_pred, show_candles

#update order inputs
#@app.callback(
#    Output('fiat-box', 'value'),
#    [Input('coin-box', 'value'), Input('price-box', 'value')])
#def update_coin_fiat(coin, price):
#        price=float(price) if price is not None else price
#        coin=float(coin) if coin is not None else coin
#        if price != None and price > 0 and coin != None and coin > 0:
#            return coin*price
#        else:
#            return dash.no_update

#update order inputs
@app.callback(
    [Output('coin-box', 'value'), Output('fiat-box', 'value')],
    [Input('coin-box', 'value'), Input('fiat-box', 'value'),
     Input('price-box', 'value')])
def update_coin_fiat(coin, fiat, price):
    ctx = dash.callback_context
    price=float(price) if price is not '' else price
    if ctx.triggered and price != '' and price >0:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if input_id == 'coin-box':
            coin=float(coin) if coin is not '' else coin
            if coin != '' and coin > 0:
                return dash.no_update, coin*price
            else:
                return dash.no_update, dash.no_update
        elif input_id == 'fiat-box':
            fiat=float(fiat) if fiat is not '' else fiat
            if fiat != '' and fiat > 0:
                return fiat/price, dash.no_update
            else:
                return dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update


#update drop-down menus
@app.callback(
    [Output('coin-choice', 'options'),
     Output('coin-add', 'options'), Output('time-add', 'options'),
     Output('coin-add', 'value'), Output('time-add', 'value')],
    [Input('graph-update', 'n_intervals'), Input('coin-add', 'value'),
     Input('time-add', 'value')])
def update_date_dropdown(n, new_pairs, new_times):
    pairs, times, pairs_all, times_all=read_write_general_config(new_pairs, new_times)
    pairs_times=[pair+"-"+time for pair in pairs for time in times]
    options_pair = [{'label': i, 'value': i} for i in pairs_times]
    options_pair_all = [{'label': i, 'value': i} for i in pairs_all]
    options_time_all = [{'label': i, 'value': i} for i in times_all]
    return options_pair, options_pair_all, options_time_all, pairs, times


#update graph
@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals'), Input('coin-choice', 'value'),
     Input('n-predictions', 'data'), Input('step-predictions', 'data'), 
     Input('length-predictions', 'data'), Input('show-candles', 'data')])
def update_graph_scatter(n, coin_time, n_pred, step_pred, len_pred, show_candles):

    #get coin data
    coin = 'XRPEUR'
    time = '1h'
    if not coin_time is None:
        coin, time = coin_time.split('-')
    df, fut, buy, sell = load_data(coin, time)

    #set historic prediction parameters
    if n_pred is None:
        n_pred=0
    if step_pred is None:
        step_pred=1
    if len_pred is None:
        len_pred=16

    #separate old futures from current future
    fut_old=np.array([])
    if len(fut.shape)>1:
        fut_old=fut[:len_pred+1,1:]
        fut=fut[:len_pred+1,0]
        if len(fut_old.shape)==1:
            fut_old=np.reshape(fut_old,(-1,1))
        fut_old=np.transpose(fut_old)
    fut_old=fut_old[:n_pred]

    #get global plot parameters
    min_x=df['Open time'].iloc[0]
    Dx=df['Open time'].iloc[1]-min_x
    last_x=df['Open time'].iloc[-1]
    max_x=last_x+Dx*16
    x_range=[last_x+Dx*i for i in range(1,len_pred+1)]
    min_y = min([min(df['Close']),min(fut)])
    max_y = max([max(df['Close']),max(fut)])
    t_vals = [i for i in np.arange(min_x,max_x,15*Dx)]
    t_text = [datetime.fromtimestamp(val/1000) for val in t_vals]
    #t_text = [str(t.hour)+":"+str(t.minute) for t in t_text]

    #past (candles)
    data=[]
    if show_candles:
        data = [go.Candlestick(
                x=[datetime.fromtimestamp(val/1000) for val in df['Open time']],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Candles')]

    #past (only line) 
    data0 = go.Scatter( 
            x=[datetime.fromtimestamp(val/1000) for val in df['Open time']], 
            y=df['Close'], 
            name='Past', 
            line=dict(color="#7fdbff", width=2),
            mode= 'lines'#+markers'
    )

    #future  
    data1 = go.Scatter( 
            x=[datetime.fromtimestamp(val/1000) for val in x_range],  
            y=fut[1:], 
            name='Future', 
            line=dict(color="#ff3300", width=2),
            mode= 'lines'#+markers'
    )

    #future old
    data1_old=[]
    for i in range(0,len(fut_old), step_pred):
        f=fut_old[i]
        data_old = go.Scatter(
                x=[datetime.fromtimestamp((val-Dx*(i+1))/1000) for val in x_range],  
                y=f[1:],
                name='Old Fut',
                line=dict(color="#ffe476", width=1),
                opacity=0.5,
                showlegend = True if i==0 else False,
                hoverinfo='skip',
                mode= 'lines'#+markers'
        )
        data1_old.append(data_old)

    #present
    data2 = go.Scatter( 
            x=[datetime.fromtimestamp(x_range[0]/1000)], 
            y=[fut[0]], 
            name='Present', 
            marker=dict(color="#00ff7b"),
            mode= 'markers'
    )

    #buy points
    data3 = go.Scatter( 
            x=[datetime.fromtimestamp(val/1000) for val in buy['time']],  
            y=buy['price'], 
            name='Buy', 
            marker=dict(color="#991200"),
            mode= 'markers'
    )

    #sell points
    data4 = go.Scatter( 
            x=[datetime.fromtimestamp(val/1000) for val in sell['time']],
            y=sell['price'], 
            name='Sell', 
            marker=dict(color="#009917"),
            mode= 'markers'
    )
  
    layout = go.Layout(
                #title=coin+' - '+time,
                xaxis_rangeslider_visible=False,
                xaxis = dict(tickmode = 'array', tickvals = t_vals, 
                             ticktext = None, gridcolor='#3d3d36'),
                #xaxis = dict(range = [min_x, max_x], gridcolor='#3d3d36'),
                yaxis = dict(range = [min_y, max_y], gridcolor='#3d3d36'),
                #width = 700,
                #height = 500,
                #paper_bgcolor = colors['background-R'],
                #plot_bgcolor = colors['background-R'],
                #font = {'color': colors['text']},
                colorway=["#7FDBFF", '#ff3300', '#00ff7b', '#991200', '#009917'],
                template='plotly_dark',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                margin={'b': 15},
                hovermode='x',
                uirevision=coin_time, #do not reset zoom setting after graph update
                autosize=True,
                title={'text': coin+' - '+time, 'font': {'color': 'white'}, 'x': 0.5},
                #showlegend=False,
    )

#    fig = go.Figure()
#    fig.add_trace(data0)
#    fig.add_trace(data1)
#    fig.add_trace(data2)
#    fig.add_trace(data3)
#    fig.add_trace(data4)
#    fig.layout = layout
#    fig.update_layout(hovermode='x unified')
#    return fig
  
    data_plot=data+[data0,data1]+data1_old+[data2,data3,data4]
    return {'data': data_plot,
            'layout' : layout} 

##################################
#run server
if __name__ == '__main__': 
    #app.run_server()
    app.run_server(debug=False, port=8080,host='0.0.0.0')
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
#output at http://localhost:8050/
#from local network: 192.168.100.5:8080 (my ip)
