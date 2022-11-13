import pandas as pd
import time
import os
import sys
import re
import math
import numpy as np
import json
import csv
import requests

data_folder="../../../data/"
base_folder=data_folder+"data_twitter/"
filename="fear&greed_index.json"

def get_fear_greed_index():
    url_request="https://api.alternative.me/fng/?limit=0&?format=json"
    #df_fear_greed = pd.read_json(url_request)
    response = requests.get(url_request)
    data = json.loads(response.text)
    with open(base_folder+filename, 'w') as f:
        json.dump(data, f)

get_fear_greed_index()
