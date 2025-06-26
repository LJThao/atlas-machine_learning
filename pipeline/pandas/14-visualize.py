#!/usr/bin/env python3
"""Visualize Module"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file
df = from_file(
    '/root/atlas-machine_learning/pipeline/pandas/data/coinbase.csv',
    ','
)


# remove Weighted_Price
df.drop(columns=['Weighted_Price'], inplace=True)

# rename Timestamp to Date and convert to datetime
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# set index to Date
df.set_index('Date', inplace=True)

# fill missing values
df['Close'] = df['Close'].ffill()

for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col] = df[col].fillna(0)

# filter from 2017 and group by day
daily_rows = df[df.index >= '2017-01-01'].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# plot all six columns
daily_rows[['High', 'Low', 'Open', 'Close',
            'Volume_(BTC)', 'Volume_(Currency)']].plot()

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('BTC Trends: 2017-2019')
plt.legend()
plt.tight_layout()
plt.show()

# print the transformed df
print(daily_rows)
