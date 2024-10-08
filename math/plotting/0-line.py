#!/usr/bin/env python3
"""Plotting y as a line graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plotting a y line graph"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    # plotting the data y with a red color line
    plt.plot(y, color="red")
    plt.xlim([0, 10])
    plt.show()
