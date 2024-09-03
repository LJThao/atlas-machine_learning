#!/usr/bin/env python3
"""Plotting a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """A stacked bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    # plotting a stacked bar graph
    columns = ["Farrah", "Fred", "Felicia"]
    rows = ["apples", "bananas", "oranges", "peaches"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    bottom = np.zeros(fruit[0].shape)
    for i in range(len(fruit)):
        plt.bar(range(len(fruit[0])), fruit[i], color=colors[i],
                bottom=bottom, width=0.5)
        bottom = bottom + fruit[i]
    plt.ylabel("Quantity of Fruit")
    plt.ylim([0, 80])
    plt.yticks(range(0, 81, 10))
    plt.xticks([0, 1, 2], labels=columns)
    plt.title("Number of Fruit per Person")
    plt.legend(rows)
    plt.show()
