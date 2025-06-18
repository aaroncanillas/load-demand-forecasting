# %% [markdown]
# ## Data Cleaning and Exploration

# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
ENERGY_DATASET = '../data/raw/energy_dataset.csv'
WEATHER_FEATURES_DATASET = '../data/raw/weather_features.csv'

df_weather = pd.read_csv(WEATHER_FEATURES_DATASET, parse_dates=['dt_iso'])
df_energy = pd.read_csv(ENERGY_DATASET, parse_dates=['time'])

# %% [markdown]
# ### Energy Dataset

# %%
df_energy.head()

# %%
# Drop columns that cannot be used
# Columns that have zero or null values, day ahead forecast, load forecast, price features

df_energy = df_energy.drop([
    'generation fossil coal-derived gas',
    'generation fossil oil shale',
    'generation fossil peat',
    'generation geothermal',
    'generation hydro pumped storage aggregated',
    'generation marine',
    'generation wind offshore',
    'forecast solar day ahead',
    'forecast wind offshore eday ahead',
    'forecast wind onshore day ahead',
    'total load forecast'],
    axis=1)

df_energy.describe().round(2)

# %%
df_energy.info()

# %%
df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
df_energy = df_energy.set_index('time')

# %%
# Validate missing and duplicate values

print(f"Number of missing values in df_energy: {df_energy.isnull().values.sum()}")
print(f"Number of duplicate values in df_energy: {df_energy.duplicated(keep='first').sum()}")

# %%
# Find the number of null values in each column

df_energy.isnull().sum(axis=0)

# %%
df_energy[df_energy.isnull().any(axis=1)].head()

# %%
# Fill null values using interpolation
df_energy.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

# %%
# Display the number of non-zero values in each column
# Existing zero values in some columns possibly due to non utilization of certain energy sources

print('Non-zero values in each column:\n', df_energy.astype(bool).sum(axis=0), sep='\n')

# %%
df_energy.head()

# %% [markdown]
# ### Weather Features Data Cleaning

# %%
df_weather.head()

# %%
# Drop columns with qualitative weather information
df_weather = df_weather.drop(['weather_main', 'weather_id', 
                              'weather_description', 'weather_icon'], axis=1)
df_weather.head()

# %%
df_weather.describe().round(2)

# Outliers existing on pressure and wind_speed

# %%
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), np.nan)
    return df

# Apply to pressure and wind_speed
df_weather = remove_outliers_iqr(df_weather, 'pressure')
df_weather = remove_outliers_iqr(df_weather, 'wind_speed')

# %%
df_weather.info()

# %%
# Convert types to float64 for consistency

def convert_dtype(df: pd.DataFrame, old_dtype: str, new_dtype: str) -> pd.DataFrame:
    for column in df.columns:
        if df[column].dtype == old_dtype:
            df[column] = df[column].astype(new_dtype)
    return df

convert_dtype(df_weather, 'int64', 'float64')
df_weather.info()

# %%
df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
df_weather = df_weather.drop(['dt_iso'], axis=1)
df_weather = df_weather.set_index('time')

df_weather.head()

# %%
print(f"Number of missing values in df_weather: {df_weather.isnull().values.sum()}")
print(f"Number of duplicate values in df_weather: {df_weather.duplicated(keep='first').sum()}")

# %%
# Average weather features by city

mean_weather_by_city = df_weather.groupby('city_name').mean(numeric_only=True)
mean_weather_by_city

# %%
cities = df_weather['city_name'].unique()
print(cities)

grouped_weather = df_weather.groupby('city_name')

for city in cities:
    print(f"Number of records for {city} in df_weather: {df_weather[df_weather['city_name'] == city].shape[0]}")

# %%
df_weather_2 = df_weather.reset_index().drop_duplicates(subset=['time', 'city_name'], 
                                                        keep='last').set_index('time')
df_weather = df_weather.reset_index().drop_duplicates(subset=['time', 'city_name'],
                                                      keep='first').set_index('time')

# %%
# Display number of rows in each data frame again

print(f"Number of records in df_energy: {df_energy.shape[0]}")

grouped_weather = df_weather.groupby('city_name')
for city in cities:
    print(f"Number of records for {city} in df_weather: {df_weather[df_weather['city_name'] == city].shape[0]}")

# %%
# Display the number of duplicates in df_weather

temp_weather = df_weather.reset_index().duplicated(subset=['time', 'city_name'], 
                                                   keep='first').sum()
print(f"Number of duplicate records in df_weather: {temp_weather}")

# %% [markdown]
# ### Cleaning outliers in 'Pressure'

# %%
df_weather.describe().round(2)


# %%
df_weather.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

# %% [markdown]
# ## Merge Datasets

# %%
df_1, df_2, df_3, df_4, df_5 = [x for _, x in df_weather.groupby('city_name')]
dfs = [df_1, df_2, df_3, df_4, df_5]

# %%
df_1.head()

# %%
df_final = df_energy

df_final = df_energy.reset_index()  # Ensure time is a column and avoid index conflicts
df_final['time'] = pd.to_datetime(df_final['time'], utc=True)

for df in dfs:
    city = df['city_name'].unique()[0]
    city_str = str(city).replace(" ", "")

    # Avoid index column duplication
    df = df.loc[:, ~df.columns.str.contains('^level_0$|^index$')]
    df = df.reset_index()

    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.add_suffix(f"_{city_str}")
    df.columns = ['time' if 'time' in col else col for col in df.columns]

    df_final = df_final.merge(df, on='time', how='outer')

    city_col = f'city_name_{city_str}'
    if city_col in df_final.columns:
        df_final.drop(columns=[city_col], inplace=True)

# %%
# Display the number of NaNs and duplicates in the final dataframe
print(f"Number of missing values in df_weather: {df_final.isnull().values.sum()}")
print(f"Number of duplicate values in df_weather: {df_final.duplicated(keep='first').sum()}")


# %%
df_final.head()


# %%
df_final.to_csv('../data/clean/df_cleaned.csv')


