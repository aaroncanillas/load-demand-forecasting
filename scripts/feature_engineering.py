# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_final = pd.read_csv('../data/clean/df_cleaned.csv')

# %%
df_final = df_final.drop(columns=['Unnamed: 0'])
df_final = df_final.set_index('time')
df_final.info()

# %%
df_final.head()

# %%
# Find the correlations between the actual load and the rest of the features

correlations = df_final.corr(method='pearson')
print(correlations['total load actual'].sort_values(ascending=False).to_string())

# %%
# Drop columns that give NaNs in their correlations with the electricity actual price.

df_final = df_final.drop(['snow_3h_Barcelona', 'snow_3h_Seville'], axis=1)

# %% [markdown]
# ## Feature Engineering

# %%
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear

    df['is_weekend'] = df.index.dayofweek >= 5
    return df

# %%
df_final.index = pd.to_datetime(df_final.index, utc=True)
df_final = create_features(df_final)

# %%
# Visualize our feature & target relationship

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_final, x='hour', y='total load actual')
ax.set_title('Total Load Actual by Hour')

# %%
# By month
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_final, x='month', y='total load actual')
ax.set_title('Total Load Actual by Month')

# %%
df_final.info()

# %%
df_final.to_csv('../data/prepared/df_prepared.csv')


