#!/bin/bash

python3 tcn_net_multi_tech_sent.py train

python3 tcn_net_multi_tech_sent.py train_final

#to get weights used for action network training 
#(they should match the validation accuracy but on the whole dataset)
python3 tcn_net_multi_tech_sent.py train_action
