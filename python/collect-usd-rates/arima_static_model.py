import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("data/processed_data.csv")
df['date'] = pd.to_datetime(df['date'])

# =========================
# 2. Train / Test
# =========================
train_size = int(len(df) * 0.8)
train = df[:train_size]
test  = df[train_size:]

train_series = train['usd_uzs']
test_series  = test['usd_uzs']

# =========================
# 3. Подбор параметров (используем тот же порядок)
# =========================
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

best_order = (4, 1, 2)  # тот же порядок, что и в rolling forecast
print(f"Порядок ARIMA: {best_order}")

# =========================
# 4. Статический прогноз (без переобучения)
# =========================
# Обучаем модель ОДИН раз на train
model = ARIMA(train_series, order=best_order)
model_fit = model.fit()

# Прогноз на весь тестовый период
predictions = model_fit.forecast(steps=len(test_series))

forecast = pd.Series(predictions.values, index=test_series.index)

# =========================
# 5. Метрики
# =========================
mae  = mean_absolute_error(test_series, forecast)
rmse = np.sqrt(mean_squared_error(test_series, forecast))
mape = np.mean(np.abs((test_series.values - forecast.values) / test_series.values)) * 100

print(f"\nMAE:  {mae:.2f} сум")
print(f"RMSE: {rmse:.2f} сум")
print(f"MAPE: {mape:.2f}%")

# =========================
# 6. Сохранение прогноза
# =========================
results_df = pd.DataFrame({
    'date':     test['date'].values,
    'actual':   test_series.values,
    'forecast': forecast.values
})
results_df.to_csv("results/arima_static_forecast.csv", index=False)
print("Прогноз сохранён в results/arima_static_forecast.csv")

# =========================
# 7. График
# =========================
plt.figure(figsize=(14, 7))
plt.plot(test['date'].values, test_series.values,
         label='Реальные значения', linewidth=1.5)
plt.plot(test['date'].values, forecast.values,
         label=f'Static Forecast ARIMA{best_order}',
         linewidth=1.5, alpha=0.8)
plt.xlabel("Дата")
plt.ylabel("Курс USD/UZS")
plt.title(f"Static Forecast ARIMA{best_order} курса USD/UZS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/arima_static_forecast.png", dpi=300)
plt.show()
print("График сохранён в plots/arima_static_forecast.png")
