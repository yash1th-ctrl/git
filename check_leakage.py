import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load without header
df = pd.read_csv('./data/UNSW_NB15_sample.csv', header=None)

# Drop non-numeric columns for correlation check
df_numeric = df.select_dtypes(include=[np.number])

# Calculate correlation with the last column (Label)
# Assuming last column is the label.
label_col_index = df.shape[1] - 1
correlations = df_numeric.corrwith(df[label_col_index])

print("Top 10 features most correlated with label:")
print(correlations.abs().sort_values(ascending=False).head(15))
