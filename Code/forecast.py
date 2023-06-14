#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.2f}'.format)


# In[2]:


file_id = "1DopC7bm_EWX_ocqbOIkME9usE5rG3pZE" # ID of the file on Google Drive
file_name = 'Customer_data_2021&2022.csv'

get_ipython().run_line_magic('run', 'download.ipynb')


# In[3]:


# File path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_dir, 'Data',file_name)

df = pd.read_csv(file_path)
df.head()


# In[7]:


df['DATE_DIM'] = pd.to_datetime(df['DATE_DIM'], format='%Y-%m-%d')

X = df.set_index('DATE_DIM').resample('d')[['RACING_TURNOVER', 'SPORT_TURNOVER', 'FOB_TURNOVER', 'PARI_TURNOVER', 'TURNOVER', 'DIVIDENDS_PAID', 'GROSS_MARGIN', 'TICKETS']].sum()
X = X[:365]
X


# In[9]:


## 2021: RACING_TURNOVER vs SPORT_TURNOVER

fig, ax = plt.subplots(2, figsize=(16,8))

ax[0].plot(X['RACING_TURNOVER'])
ax[0].set_title('2021 RACING_TURNOVER')

ax[1].plot(X['SPORT_TURNOVER'])
ax[1].set_title('2021 SPORT_TURNOVER')

plt.show()


# In[10]:


X_racing = X['RACING_TURNOVER']
X_sport = X['SPORT_TURNOVER']


# # Racing

# In[105]:


X_train = X_racing[:181]
X_test = X_racing[181:212]

exp_smth = ExponentialSmoothing(X_train, seasonal_periods=7, trend = "add", seasonal = "mul")
result = exp_smth.fit()

forecast = result.predict(start=X_train.index[-1] + pd.DateOffset(1), end=X_train.index[-1] + pd.DateOffset(31))


# In[106]:


alpha = result.params['smoothing_level']
alpha


# In[107]:


beta = result.params['smoothing_trend']
beta


# In[108]:


gamma = result.params['smoothing_seasonal']
gamma


# In[123]:


plt.figure(figsize=(16,4))

plt.plot(X_train)
plt.plot(forecast)
plt.plot(X_test)

plt.legend(['train', 'predicted', 'real'], loc = 'upper left')

plt.show()


# In[110]:


plt.figure(figsize=(16,4))

plt.plot(forecast, c='orange')
plt.plot(X_test, c='green', label='real')
plt.legend(['predicted', 'real'])

plt.show()


# In[111]:


plt.figure(figsize=(16,4))

diff = (X_test - forecast)
plt.plot(diff)


# In[112]:


MSE = np.linalg.norm(diff)/62
MSE


# In[113]:


plt.figure(figsize=(16,4))

diff = (X_test - forecast) / X_test * 100
plt.plot(diff)
plt.axhline(y = 0.0, color = 'r', linestyle = '-')
plt.ylim(-10,35)


# In[119]:


plt.figure(figsize=(16,4))
X = df.set_index('DATE_DIM').resample('W')['RACING_TURNOVER'].sum()
X = X[:52]
plt.plot(X)


# incease in August

# In[131]:


X_train = X_racing[:120]
X_test = X_racing[120:150]

exp_smth = ExponentialSmoothing(X_train, seasonal_periods=7, trend = "add", seasonal = "mul")
result = exp_smth.fit()

forecast = result.predict(start=X_train.index[-1] + pd.DateOffset(1), end=X_train.index[-1] + pd.DateOffset(30))


# In[132]:


plt.figure(figsize=(16,4))

plt.plot(X_train)
plt.plot(forecast)
plt.plot(X_test)

plt.legend(['train', 'predicted', 'real'], loc = 'upper left')

plt.show()


# In[135]:


plt.figure(figsize=(16,4))

plt.plot(forecast, c='orange')
plt.plot(X_test, c='green', label='real')
plt.legend(['predicted', 'real'])

plt.show()


# In[137]:


plt.figure(figsize=(16,4))

diff = (X_test - forecast)
plt.plot(diff)


# In[138]:


MSE = np.linalg.norm(diff)/62
MSE


# In[141]:


plt.figure(figsize=(16,4))

diff = (X_test - forecast) / X_test * 100
plt.plot(diff)
plt.axhline(y = 0.0, color = 'r', linestyle = '-')
plt.ylim(-30,10)


# In[ ]:





# In[ ]:





# In[ ]:




