#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  test_rating_pred.py
    Author:     tigong
    Date:       19-2-13
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
import os
import sys
import argparse
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from resys_model.rating_prediction.mf import MF
from resys_model.utils.load_data.load_data_rating import load_data_rating


def parse_args():
    parser = argparse.ArgumentParser(description="Rating_Predict_Test")
    parser.add_argument("--model", choices=['MF', 'NNMF', 'FM', 'NFM', 'AFM'], default="MF")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_factors", type=int, default=10)
    parser.add_argument("--display_step", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--reg_rate", type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size

    train_data, test_data, n_user, n_item = load_data_rating(path="../data/ml100k/movielens_100k.dat",
                                                             header=["user_id", "item_id", "rating", "t"],
                                                             test_size=0.1, sep="\t")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        model = None

        if args.model == "MF":
            model = MF(sess, n_user, n_item, batch_size=batch_size)

        if model is not None:
            model.build_graph()
            model.execute(train_data, test_data)
