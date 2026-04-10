
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# =============================
# BAI 1 - DOANH THU SIEU THI
# =============================
supermarket = pd.read_csv('ITA105_Lab_5_Supermarket.csv')
supermarket['date'] = pd.to_datetime(supermarket['date'])
supermarket = supermarket.set_index('date').sort_index()

# Dien missing values
supermarket['revenue'] = supermarket['revenue'].ffill().bfill().interpolate()

# Tao dac trung
supermarket['year'] = supermarket.index.year
supermarket['month'] = supermarket.index.month
supermarket['quarter'] = supermarket.index.quarter
supermarket['day_of_week'] = supermarket.index.day_name()
supermarket['type_day'] = np.where(supermarket.index.dayofweek >= 5, 'Weekend', 'Weekday')

# Tong doanh thu theo thang, tuan
monthly_revenue = supermarket['revenue'].resample('ME').sum()
weekly_revenue = supermarket['revenue'].resample('W').sum()

# Trend va seasonality
supermarket['rolling_30'] = supermarket['revenue'].rolling(30).mean()
decomp_supermarket = seasonal_decompose(supermarket['revenue'], model='additive', period=7)

# =============================
# BAI 2 - WEB TRAFFIC
# =============================
web = pd.read_csv('ITA105_Lab_5_Web_traffic.csv')
web['datetime'] = pd.to_datetime(web['datetime'])
web = web.set_index('datetime').sort_index().asfreq('h')

# Noi suy du lieu thieu
web['visits'] = web['visits'].interpolate(method='linear')

# Tao dac trung
web['hour'] = web.index.hour
web['day_of_week'] = web.index.day_name()

hourly_avg = web.groupby('hour')['visits'].mean()

# Daily & weekly seasonality
decomp_web_daily = seasonal_decompose(web['visits'], model='additive', period=24)
decomp_web_weekly = seasonal_decompose(web['visits'], model='additive', period=168)

# =============================
# BAI 3 - GIA CO PHIEU
# =============================
stock = pd.read_csv('ITA105_Lab_5_Stock.csv')
stock['date'] = pd.to_datetime(stock['date'])
stock = stock.set_index('date').sort_index().asfreq('D')

# Fill ngay nghi giao dich
stock['close_price'] = stock['close_price'].ffill()

# Rolling mean
stock['ma7'] = stock['close_price'].rolling(7).mean()
stock['ma30'] = stock['close_price'].rolling(30).mean()

# Seasonality theo thang
monthly_return = stock['close_price'].pct_change().groupby(stock.index.month).mean() * 100

# =============================
# BAI 4 - SAN XUAT CONG NGHIEP
# =============================
production = pd.read_csv('ITA105_Lab_5_Production.csv')
production['week_start'] = pd.to_datetime(production['week_start'])
production = production.set_index('week_start').sort_index().asfreq('W-SUN')

# Dien missing values
production['production'] = production['production'].interpolate().ffill().bfill()

# Tao dac trung
production['week'] = production.index.isocalendar().week.astype(int)
production['quarter'] = production.index.quarter
production['year'] = production.index.year

# Trend va decomposition
production['trend_12w'] = production['production'].rolling(12).mean()
quarter_avg = production.groupby('quarter')['production'].mean()
decomp_production = seasonal_decompose(production['production'], model='additive', period=13)

print("Lab 5 da xu ly xong du lieu va tao day du dac trung.")
