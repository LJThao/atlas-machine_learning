#!/usr/bin/env python3
"""Moving Average Function"""
import numpy as np


def moving_average(data, beta):
    """Function that calculates the weighted moving average of a data set:

    data = the list of data to calculate the moving average of
    beta = the weight used for the moving average

    """