import pandas as pd
import numpy as np
import json
import csv

data = pd.read_csv("raw/lexicon_crypto.csv") 
my_data=data.to_numpy()
#my_data=my_data.lower()
for i in range(len(my_data)):
    my_data[i][0] = my_data[i][0].lower()
    my_data[i][1] = float(my_data[i][1])*4
print(my_data[0])

#d = data.to_dict()
d=dict(my_data)

with open("lexicon_crypto.json", "w") as dic:
    json.dump(d, dic, indent=0)


