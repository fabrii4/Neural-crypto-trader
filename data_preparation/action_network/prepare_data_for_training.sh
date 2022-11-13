#!/bin/bash

#@TODO optimize these files to run from a python file with a single list of coins etc.

#- TRAIN PREDICTION DATASET
#- SET THE CORRECT WEIGHT FILE IN create_prediction_dataset.py

#get historical data from binance
python3 create_prediction_dataset.py
#prepare technical indicators
python3 create_q-value_dataset.py
