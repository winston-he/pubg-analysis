#! usr/bin/python
# -*- coding:utf-8 -*-
# @Author: winston he
# @File: weapon.py
# @Time: 2020-07-14 15:21
# @Email: winston.wz.he@gmail.com
# @Desc:
import pickle
from collections import defaultdict
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt


CHUNKSIZE = 1000000


weapons = defaultdict(int)

distances = defaultdict(float)

data_size = 0

for i in range(2):
    data = pd.read_csv("kill_match_stats_final_{}.csv".format(i), chunksize=CHUNKSIZE)
    for chunk in data:
        chunk["count"] = 1
        # chunk["distance"] = sqrt((chunk["killer_position_x"]-chunk["victim_position_x"])**2 + (chunk["killer_position_y"]-chunk["victim_position_y"])**2)

        chunk["x_distance"] = chunk["killer_position_x"] - chunk["victim_position_x"]
        chunk["x_distance"] **= 2
        chunk["y_distance"] = chunk["killer_position_y"] - chunk["victim_position_y"]
        chunk["y_distance"] **= 2
        chunk["distance"] = chunk["y_distance"] + chunk["x_distance"]
        chunk["distance"] = chunk["distance"].apply(lambda x: sqrt(x))

        result = chunk[["killed_by", "count", "distance"]].groupby("killed_by").agg({"count": "sum", "distance": "sum"})
        result.reset_index(inplace=True)
        for r in result.values:
            weapons[r[0]] += r[1]
            distances[r[0]] += r[2]
        data_size += len(chunk.values)

for k in weapons.keys():
    distances[k] = distances[k] / weapons[k]

with open("weapons.pickles", "wb") as f:
    pickle.dump(weapons, f)

with open("distance.pickles", "wb") as f:
    pickle.dump(distances, f)

X = [k for k in weapons.keys()]
Y = [v for v in weapons.values()]



plt.bar(x=X, height=Y, label="武器排行", alpha=0.8)
print("Total data size: ", data_size)


plt.show()