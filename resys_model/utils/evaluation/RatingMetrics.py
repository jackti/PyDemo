#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  RatingMetrics.py
    Author:     tigong
    Date:       19-2-13
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""

import numpy as np


def RMSE(error, num):
    return np.sqrt(error * 1.0 / num)


def MAE(error, num):
    return error * 1.0 / num
