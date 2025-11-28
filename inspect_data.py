import pandas as pd
import numpy as np

# Load without header
df = pd.read_csv('./data/UNSW_NB15_sample.csv', header=None)

print("Shape:", df.shape)
print("Last 5 columns sample:")
print(df.iloc[:5, -5:])

print("\nUnique values in last column (index -1):")
print(df.iloc[:, -1].unique())

print("\nUnique values in second to last column (index -2):")
print(df.iloc[:, -2].unique())

print("\nUnique values in third to last column (index -3):")
print(df.iloc[:, -3].unique())
