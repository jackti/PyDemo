#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  load_data_rating.py
    Author:     tigong
    Date:       19-2-18
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def load_data_rating(path="../data/ml100k/movielens_100k.dat",
                     header=["user_id", "item_id", "rating", "category"],
                     test_size=0.1, sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine="python")
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    print("load data finished. number of users:", n_users, "number of items:", n_items)

    return train_matrix.todok(), test_matrix.todok(), n_users, n_items
