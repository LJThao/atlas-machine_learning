#!/usr/bin/env python3
"""Moving Average Function"""
import numpy as np


def moving_average(data, beta):
    """Function that calculates the weighted moving average of a data set:

    data = the list of data to calculate the moving average of
    beta = the weight used for the moving average

    """
    # initializes the avg weight to 0, and an empty list
    w_avg_list = []
    w_avg = 0
    # iterate each data point, update the weighted avg, apply bias correction
    # of the moving avg, then build a list of all the moving avg
    for i in range(len(data)):
        w_avg = (beta * w_avg) + (1 - beta) * data[i]
        bias_correction_avg = w_avg / (1 - (beta ** (i + 1)))
        w_avg_list.append(bias_correction_avg)

    # returns a list containing the moving averages of data
    return (w_avg_list)
