#! usr/bin/python
# -*- coding:utf-8 -*-
# @Author: winston he
# @File: hitmap.py
# @Time: 2020-07-10 15:01
# @Email: winston.wz.he@gmail.com
# @Desc:
import code
from random import randint

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
import numpy as np
import os
# ERANGEL MIRAMAR

data = pd.read_csv("kill_match_stats_final_0.csv", iterator=True)
max_coor = 8 * 1e5


EPS = 3900
MIN_SAMPLES = 220

CHUNKSIZE = 2000

def coodinate_shift(p):
    return max_coor - p

#
# def coodinate_scale(p):
#     return p / max_coor


chunk = data.get_chunk(CHUNKSIZE)
X = y = None
hex_range = "0123456789abcdef"

# model training
for _ in range(500):
    chunk["victim_position_y"] = chunk["victim_position_y"].apply(func=coodinate_shift)
    chunk = chunk[chunk.map == "ERANGEL"]

    # #FFCC33
    X = np.reshape(chunk["victim_position_x"].values, (-1, 1)) if X is None else \
        np.vstack((X, np.reshape(chunk["victim_position_x"].values, (-1, 1))))

    y = np.reshape(chunk["victim_position_y"].values, (-1, 1)) if y is None else \
        np.vstack((y, np.reshape(chunk["victim_position_y"].values, (-1, 1))))
    chunk = data.get_chunk(CHUNKSIZE)

total_data_count = X.shape[0]
print(total_data_count)

training_data = np.array(list(zip(X, y))).reshape(-1, 2)
eps_range = list(range(3500, 4000, 200))
sample_range = list(range(100, 300, 25))

# for e in eps_range:
#     for s in sample_range:
model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
model.fit(training_data)

# visualization
group_size = max(model.labels_)
# if group_size <= 10:
#     print("Break One s:{} e:{} max:{}".format(e, s, max(model.labels_)))
#     break
#
# if group_size > 100:
#     print("Continue One s:{} e:{} max:{}".format(e, s, max(model.labels_)))
#     continue

for i in range(group_size):
    color = "#" + "".join([hex_range[randint(0, 15)] for _ in range(6)])
    curr_x = [x for idx, x in enumerate(X) if model.labels_[idx] == i]

    curr_y = [y for idx, y in enumerate(y) if model.labels_[idx] == i]
    plt.scatter(curr_x, curr_y, marker='o', color=[color, ], s=15,
                alpha=0.15, cmap='Spectral_r')
    plt.text(curr_x[0], curr_y[0], "{:.2f}%".format(len(curr_x) * 100 / total_data_count))

plt.savefig(os.path.join("results", "{}_{}_{}.png".format(EPS, MIN_SAMPLES, max(model.labels_))))
# plt.clf()
plt.show()