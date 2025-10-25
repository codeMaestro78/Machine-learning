import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE=True
except:
    XGBOOST_AVAILABLE=False

def add_technical_indicators(df):
  df=df.copy()
  df.columns=[col[0] if isinstance(col,tuple) else col for col in df.columns]
  df['return']=df['Close'].pct_change()

  # moving average
  df['ma5']=df['Close'].rolling(window=5).mean()
  df['ma10']=df['Close'].rolling(window=10).mean()
  df['ma21']=df['Close'].rolling(window=21).mean()

  # ema
  df['ema12']=df['Close'].ewm(span=12,adjust=False).mean()
  df['ema26']=df['Close'].ewm(span=26,adjust=False).mean()

  # macd
  df['macd']=df['ema12']-df['ema26']

  # bb
  df['bb_std']=df['Close'].rolling(window=20).std()
  df['bb_upper']=df['ma21']+2*df['bb_std']
  df['bb_lower']=df['ma21']-2*df['bb_std']

  # momentum
  df['mom_5']=df['Close']-df['Close'].shift(5)
  df['mom_10']=df['Close']-df['Close'].shift(10)
  df['mom_21']=df['Close']-df['Close'].shift(21)

  # rsi(classic)
  # The RSI measures the speed at which prices change and the magnitude of those changes.
  delta=df['Close'].diff()
  up=delta.clip(lower=0)
  down=-1*delta.clip(upper=0)
  roll_up=up.ewm(span=14,adjust=False).mean()
  roll_down=down.ewm(span=14,adjust=False).mean()
  rs=roll_up/(roll_down+1e-9)
  df['rsi']=100.0-(100.0/(1.0+rs))

  # volume feature
  df['vol_10']=df['Volume'].rolling(window=10).mean()
  df['vol_ratio']=df['Volume'] / (df['vol_10'] + 1e-9)

  # prev days close
  for lag in [1,2,3,5,10]:
    df[f'lag_close_{lag}']=df['Close'].shift(lag)

  return df


# load data
TICKER="TSLA"
START='2020-01-01'
END=datetime.today().strftime('%Y-%m-%d')


print(f"Downloading {TICKER} from {START} to {END}")
raw=yf.download(TICKER,start=START,end=END,progress=False)
raw.reset_index(inplace=True)
raw.head()

#  feature engineering + target for regression

df=raw.copy()
df=add_technical_indicators(df)
df.head(5)

df['target']=df['Close'].shift(-1)

intial_len=len(df)
df.dropna(inplace=True)
print(f"Dropped {intial_len-len(df)} rows with nans - remaining: {len(df)}")


features=[c for c in df.columns if c not in ['Date', 'Adj Close', 'target']]
X=df[features].copy()
y=df['target'].copy()

split_idx=int(len(df)*0.8)
X_train,X_test=X.iloc[:split_idx],X.iloc[split_idx:]
y_train,y_test=y.iloc[:split_idx],y.iloc[split_idx:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# Scaling
# we will scale only the feature not target
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

joblib.dump(scaler,'scaler.joblib')

# baseline model

lr=LinearRegression()
lr.fit(X_train_scaled,y_train)
y_pred_lr=lr.predict(X_test_scaled)

# metric function
def  regression_metrics(y_true,y_pred):
  mae=mean_absolute_error(y_true,y_pred)
  rmse=root_mean_squared_error(y_true,y_pred)
  mse=mean_squared_error(y_true,y_pred)
  r2=r2_score(y_true,y_pred)
  return {"MAE":mae,"RMSE":rmse,"MSE":mse,"R2":r2}
print("Linear Regression")
print(regression_metrics(y_test,y_pred_lr))


# random forest regressor
rf=RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42, n_jobs=-1)
rf.fit(X_train_scaled,y_train)
y_pred_rf=rf.predict(X_test_scaled)
print("Random Forest metrics:", regression_metrics(y_test, y_pred_rf))


# xgboost

if XGBOOST_AVAILABLE:
  xgbr=xgb.XGBRegressor(objective='reg:squarederror',n_estimators=500,learning_rate=0.05,max_depth=5,random_state=42)
  xgbr.fit(X_train_scaled,y_train,eval_set=[(X_test_scaled,y_test)],verbose=False)
  y_pred_xgb = xgbr.predict(X_test_scaled)
  print("XGBoost metrics:", regression_metrics(y_test, y_pred_xgb))
else:
  print("XGBoost not available — skip")



# visulize

plt.figure(figsize=(14,6))
plt.plot(df['Date'].iloc[split_idx:],y_test.values,label='Actual (Next close)',linewidth=1)
plt.plot(df['Date'].iloc[split_idx:],y_pred_lr,label='Linear Regression',linewidth=1)
plt.plot(df['Date'].iloc[split_idx:],y_pred_rf,label='Random Forest',linewidth=3)
if XGBOOST_AVAILABLE:
  plt.plot(df['Date'].iloc[split_idx:],y_pred_xgb,label='XGBoost',linewidth=3)
plt.legend()
plt.title(f"Actual vs Predicted Next-Day Close — {TICKER}")
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# residual diagnostics

residuals=y_test.values-y_pred_rf
plt.figure(figsize=(10,4))
plt.scatter(y_pred_rf,residuals,alpha=0.5)
plt.hlines(0,y_pred_rf.min(),y_pred_rf.max(),color='red',linestyles='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted (Random Forest)')
plt.show()


# feature importance (rf)

importances=rf.feature_importances_
feat_imp=pd.Series(importances,index=X_train.columns).sort_values(ascending=False)
print(feat_imp.head(21))


plt.figure(figsize=(8,6))
feat_imp.head(15).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 15 Feature Importances (RF)')
plt.show()

joblib.dump(rf, 'rf_stock_model.joblib')
print('Saved Random Forest model to rf_stock_model.joblib and scaler to scaler.joblib')
