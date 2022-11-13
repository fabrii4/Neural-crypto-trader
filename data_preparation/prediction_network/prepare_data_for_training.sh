#!/bin/bash

#@TODO optimize these files to run from a python file with a single list of coins etc.

#get historical data from binance
echo "-----------------------------------------------"
echo "Get historical data"
echo "-----------------------------------------------"
python3 get_data.py
#prepare technical indicators
echo "-----------------------------------------------"
echo "Prepare technical indicators"
echo "-----------------------------------------------"
python3 prepare_technical_ind.py

cd sentiment_analysis/
#get twitter posts
echo "-----------------------------------------------"
echo "Get twitter data"
echo "-----------------------------------------------"
python3 get_coin_twitter_comments.py
#get reddit submissions
echo "-----------------------------------------------"
echo "Get reddit submissions"
echo "-----------------------------------------------"
python3 get_coin_reddit_submissions.py
#get fear&greed index
echo "-----------------------------------------------"
echo "Get fear greed index"
echo "-----------------------------------------------"
python3 get_fear-greed.py
#get lunar data
echo "-----------------------------------------------"
echo "Get lunar data"
echo "-----------------------------------------------"
python3 get_sentiment_lunar.py
#compute twitter sentiments
echo "-----------------------------------------------"
echo "Prepare twitter sentiment indicators"
echo "-----------------------------------------------"
python3 vader_twitter.py
#compute reddit sentiments
echo "-----------------------------------------------"
echo "Prepare reddit sentiment indicators"
echo "-----------------------------------------------"
python3 vader_reddit.py

cd ..
#combine technical sentiment indicators
echo "-----------------------------------------------"
echo "Combine technical sentiment indicators"
echo "-----------------------------------------------"
python3 prepare_technical_sentiment_ind.py
