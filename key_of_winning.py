#! usr/bin/python
# -*- coding:utf-8 -*-
# @Author: winston he
# @File: key_of_winning.py
# @Time: 2020-07-13 13:48
# @Email: winston.wz.he@gmail.com
# @Desc:

# training and testing a multi-class decision tree model
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler


CHUNKSIZE = 100000
FEATURES = ["player_assists", "player_dbno", "player_dist_ride", "player_kills", "player_dmg"]

def aggregate(data):
    res = data[["game_size", "match_id", "party_size", "player_assists", "player_dbno",
                       "player_dist_ride", "player_kills", "team_id", "team_placement", "player_dmg"]].groupby(
        ["match_id", "team_id"]).agg(
        {
            "game_size": "mean",
            "party_size": "mean",
            "player_assists": "sum",
            "player_dbno": "sum",
            "player_dist_ride": "sum",
            "player_kills": "sum",
            "player_dmg": "sum",
            "team_placement": "mean",
        }
    )
    res.reset_index(inplace=True)
    return res


# TRAINING
data = pd.read_csv("agg_match_stats_0.csv", iterator=True, chunksize=CHUNKSIZE)


# res = aggregate(res)
#
# res.to_csv("test.csv")
res = pd.DataFrame()

for _ in range(50):
    chunk = data.get_chunk()
    chunk = chunk[chunk.party_size != 1]
    curr_res = aggregate(chunk)
    res = pd.concat([res, curr_res]) if len(res.values) else curr_res
    res = aggregate(res)
    res["placement"] = res["team_placement"] / res["game_size"]
    res["placement"] = res["placement"].apply(func=lambda x: 0 if x < 0.25 else 1 if x < 0.5 \
        else 2 if x < 0.75 else 3)

# res.drop(["team_placement", "game_size", "party_size", "match_id", "team_id"], axis=1, inplace=True)
scaler = StandardScaler()

transformed_res = scaler.fit_transform(res[FEATURES])
# res[FEATURES] = transformed_res

print("GOT ALL THE DATA: length is: ", len(res.values))
model = tree.DecisionTreeClassifier()
model = model.fit(X=transformed_res, y=res["placement"])


# TESTING
data = pd.read_csv("agg_match_stats_1.csv", iterator=True, chunksize=CHUNKSIZE)
res = pd.DataFrame()
for _ in range(5):
    chunk = data.get_chunk()
    chunk = chunk[chunk.party_size != 1]
    curr_res = aggregate(chunk)
    res = pd.concat([res, curr_res]) if len(res.values) else curr_res
    res = aggregate(res)
    res["placement"] = res["team_placement"] / res["game_size"]
    res["placement"] = res["placement"].apply(func=lambda x: 0 if x < 0.25 else 1 if x < 0.5 \
        else 2 if x < 0.75 else 3)

scaler = StandardScaler()
transformed_res = scaler.fit_transform(res[FEATURES])
# res[FEATURES] = transformed_res
predictions = model.predict(X=transformed_res)



count = 0

for i in range(len(res["placement"])):
    if predictions[i] == res["placement"][i]:
        count += 1

print("Precision is: {}".format(count / len(res["placement"])))

for f in zip(FEATURES, model.feature_importances_):
    print("Feature: {}, Importance: {}".format(f[0], f[1]))