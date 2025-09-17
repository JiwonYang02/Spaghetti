import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import numpy as np

# File path
file_path = 'wn_route.csv'

# Load data
df_route = pd.read_csv(file_path)

# Convert to DateTime format
df_route['datetime'] = pd.to_datetime(df_route['datetime'], format='%Y%m%d_%H%M')

# Set index to datetime
df_route.set_index('datetime', inplace=True)

# STL decomposition: period = 6*24=144 (10-min interval * 6 = 1 hour, 1 day = 144 points)
stl = STL(df_route['pctTC'], period=144)
res = stl.fit()

# Predicted = trend + seasonal
df_route['predicted'] = res.trend + res.seasonal

# Calculate Z-score (based on residuals)
residual = df_route['pctTC'] - df_route['predicted']
z_scores = (residual - residual.mean()) / residual.std()
df_route['z_score'] = z_scores

# Detect anomalies
df_route['anomaly'] = df_route['z_score'] > 2.326

# Extract anomalies only
anomaly_df = df_route[df_route['anomaly'] == True].copy()

# Restore datetime as a column
anomaly_df.reset_index(inplace=True)

