import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# 종목 심볼 및 현재 날짜 설정
symbol = input("종목 : ")
start_date = '2020-01-01'
current_date = datetime.now().strftime('%Y-%m-%d')

# 최근 1년간의 데이터 가져오기
data = yf.download(symbol, start=start_date, end=current_date)

# 주가 데이터 전처리
data = data['Close'].to_frame()
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)

# 이동평균(Moving Average) 계산
window_size = 10
data['MA'] = data['Close'].rolling(window=window_size).mean()

# 데이터 시각화
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Closing Price')
plt.plot(data['Date'], data['MA'], label=f'{window_size}-day Moving Average')
plt.title(f'{symbol} Stock Price and Moving Average')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# 특성 생성
data['MA_diff'] = data['MA'].diff()
data.dropna(inplace=True)

# 특성과 레이블 생성
X = data[['MA', 'MA_diff']].values
y = data['Close'].values

# 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 훈련 데이터와 테스트 데이터로 나누기 (현재는 단일 날짜 데이터이므로 생략 가능)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 훈련
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 예측
y_pred = model.predict(X_scaled)

# 성능 측정
mse = mean_squared_error(y, y_pred)

print(f'Mean Squared Error: {mse}')

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], y, label='Actual Prices', color='blue')
plt.plot(data['Date'], y_pred, label='Predicted Prices', linestyle='dashed', color='red')
plt.title(f'{symbol} Stock Price Prediction with Random Forest')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# # 예측할 날짜 설정
# if datetime.today().weekday() == 4:
#     prediction_days = 3
# elif datetime.today().weekday() == 5:
#     prediction_days = 2
# else:
#     prediction_days = 1

# prediction_date = (pd.to_datetime(current_date) + pd.DateOffset(days=prediction_days)).strftime('%Y-%m-%d')

# # 특성과 레이블 선택
# X_linear = data['MA'].values.reshape(-1, 1)  # 'MA' 열을 2D 배열로 변환
# y_linear = data['Close'].values

# # 선형 회귀 모델 생성 및 훈련
# linear_model = LinearRegression()
# linear_model.fit(X_linear, y_linear)

# # 훈련된 모델의 계수 확인
# theta_0 = linear_model.intercept_
# theta_1 = linear_model.coef_[0]
# predict_close = lambda x: theta_1 * x + theta_0
# #print(f'선형 회귀 모델: y = {theta_0:.2f} + {theta_1:.2f} * x')

# plt.figure()
# plt.plot(data['MA'], data['Close'], 'r.', label='The given data')
# plt.plot(data['MA'], predict_close(data['MA']), 'b-', label='Prediction')
# plt.xlabel('MSE')
# plt.ylabel('Close')
# plt.grid()
# plt.legend()
# plt.show()

# # 예측 결과 출력
# print(f'최근 주가: {y[-1]}, 예측 주가: {y_pred[-1]}')
# #print(f'예측한 다음 날의 주가: {predict_close(prediction_date)}')
