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
# 3. Подбор параметров
# =========================
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

auto_model = auto_arima(
    train_series,
    seasonal=False,
    stepwise=True,
    information_criterion='aic',
    trace=True
)
best_order = auto_model.order
print(f"Лучший порядок ARIMA: {best_order}")

# =========================
# 4. Rolling Forecast
# =========================
history     = list(train_series)   # начальная история = весь train
predictions = []

print(f"\nНачинаем rolling forecast ({len(test_series)} шагов)...")

for i, real_value in enumerate(test_series):

    # Обучаем модель на текущей истории
    model     = ARIMA(history, order=best_order)
    model_fit = model.fit()

    # Прогноз только на 1 шаг вперёд
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)

    # Добавляем РЕАЛЬНОЕ значение в историю (не прогноз!)
    history.append(real_value)

    # Прогресс каждые 50 шагов
    if i % 50 == 0:
        print(f"  Шаг {i}/{len(test_series)}  прогноз={yhat:.2f}  реальное={real_value:.2f}")

forecast = pd.Series(predictions, index=test_series.index)

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
results_df.to_csv("results/arima_forecast.csv", index=False)
print("Прогноз сохранён в results/arima_forecast.csv")

# =========================
# 7. График
# =========================
plt.figure(figsize=(14, 7))
plt.plot(test['date'].values, test_series.values,
         label='Реальные значения', linewidth=1.5)
plt.plot(test['date'].values, forecast.values,
         label=f'Rolling Forecast ARIMA{best_order}',
         linewidth=1.5, alpha=0.8)
plt.xlabel("Дата")
plt.ylabel("Курс USD/UZS")
plt.title(f"Rolling Forecast ARIMA{best_order} курса USD/UZS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/arima_forecast.png", dpi=300)
plt.show()