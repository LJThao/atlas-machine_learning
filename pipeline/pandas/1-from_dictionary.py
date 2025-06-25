#!/usr/bin/env python3
"""From Dictionary Module"""
import pandas as pd


# create df with labeled rows and columns
df = pd.DataFrame({
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}, index=['A', 'B', 'C', 'D'])
