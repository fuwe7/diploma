import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("data/processed_data.csv")
df['date'] = pd.to_datetime(df['date'])

# =========================
# 2. Признаки и целевая переменная
# =========================
features = ['lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7']

X = df[features]
y = df['usd_uzs']

# =========================
# 3. Train / Test
# =========================
train_size  = int(len(df) * 0.8)
X_train     = X[:train_size]
X_test      = X[train_size:]
y_train     = y[:train_size]
y_test      = y[train_size:]
dates_test  = df['date'][train_size:]

# =========================
# 4. Обучение модели
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("Модель обучена.")

# =========================
# 5. Прогноз
# =========================
# RF использует реальные лаги из processed_data.csv —
# каждое предсказание уже является 1-шаговым (аналог rolling forecast)
predictions = model.predict(X_test)
print(f"Выполнено {len(predictions)} предсказаний.")

# =========================
# 6. Метрики
# =========================
mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test.values - predictions) / y_test.values)) * 100

print(f"\nMAE:  {mae:.2f} сум")
print(f"RMSE: {rmse:.2f} сум")
print(f"MAPE: {mape:.2f}%")

# =========================
# 7. Сохранение прогноза
# =========================
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame({
    'date':     dates_test.values,
    'actual':   y_test.values,
    'forecast': predictions
})
results_df.to_csv("results/rf_forecast.csv", index=False)
print("Прогноз сохранён в results/rf_forecast.csv")

# =========================
# 8. График
# =========================
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(14, 7))
plt.plot(dates_test.values, y_test.values,
         label='Реальные значения', linewidth=1.5)
plt.plot(dates_test.values, predictions,
         label='Прогноз Random Forest', linewidth=1.5, alpha=0.8)
plt.xlabel("Дата")
plt.ylabel("Курс USD/UZS")
plt.title("Прогноз Random Forest курса USD/UZS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/rf_forecast.png", dpi=300)
plt.show()