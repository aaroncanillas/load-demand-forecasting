# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv('../data/prepared/df_prepared.csv')
df = df.set_index('time')
df.info()

# %%
X, y = df.drop('total load actual', axis=1), df['total load actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


print(X.shape)
print(y.shape)
print(y_train.isna().sum())
print(y_test.isna().sum())

# %% [markdown]
# ## Random Forest Regressor Model

# %%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=1)
model.fit(X_train, y_train)

# %% [markdown]
# ## XGBoost Regressor Model

# %%
# Learning rate, pwede mag hyperparmeter tuning? grid search siguro

reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True)

# %%
pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance']).sort_values(by='importance', ascending=False)

# %%
y_pred_rf = model.predict(X_test)
y_pred_xgb = reg.predict(X_test)

# Metrics
metrics = {
    'Model': ['Random Forest', 'XGBoost'],
    'MAE': [
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_xgb)
    ],
    'MAPE': [
        mean_absolute_percentage_error(y_test, y_pred_rf) * 100,
        mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
    ],
    'R²': [
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb)
    ]
}


# %%
metrics_df = pd.DataFrame(metrics)
metrics_df


