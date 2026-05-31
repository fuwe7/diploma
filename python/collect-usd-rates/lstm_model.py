import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Фиксируем случайность для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("data/processed_data.csv")
df['date'] = pd.to_datetime(df['date'])

series = df['usd_uzs_scaled'].values
print("Размер ряда:", len(series))

# =========================
# 2. Создание последовательностей
# =========================
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 10  # модель видит 10 дней назад для прогноза следующего

X, y = create_sequences(series, WINDOW_SIZE)
dates = df['date'].iloc[WINDOW_SIZE:].reset_index(drop=True)

# LSTM ожидает трёхмерный массив: (образцы, шаги, признаки)
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 3. Train / Test
# =========================
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test  = X[train_size:]
y_train = y[:train_size]
y_test  = y[train_size:]

dates_test = pd.to_datetime(dates[train_size:])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# =========================
# 4. Построение модели
# =========================
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# =========================
# 5. Обучение
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 6. График обучения
# =========================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Кривая обучения LSTM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/lstm_train_graph.png', dpi=300)
plt.show()

# =========================
# 7. Прогноз
# =========================
predictions = model.predict(X_test)

# =========================
# 8. Обратная нормализация
# =========================
# Воссоздаём скейлер точно так же как в data-preparation.py
# — только на train части, без утечки тестовых данных
df_scaler    = pd.read_csv("data/processed_data.csv")
train_size_s = int(len(df_scaler) * 0.8)

scaler = MinMaxScaler()
scaler.fit(df_scaler[['usd_uzs']][:train_size_s])

predictions_real = scaler.inverse_transform(predictions)
y_test_real      = scaler.inverse_transform(y_test.reshape(-1, 1))

# =========================
# 9. Метрики
# =========================
mae  = mean_absolute_error(y_test_real, predictions_real)
rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100

print(f"\nMAE:  {mae:.2f} сум")
print(f"RMSE: {rmse:.2f} сум")
print(f"MAPE: {mape:.2f}%")

# =========================
# 10. Сохранение прогноза
# =========================
results_df = pd.DataFrame({
    'date':     dates_test.values,
    'actual':   y_test_real.flatten(),
    'forecast': predictions_real.flatten()
})
results_df.to_csv('results/lstm_forecast.csv', index=False)
print("Прогноз сохранён в results/lstm_forecast.csv")

# =========================
# 11. График прогноза
# =========================
plt.figure(figsize=(14, 7))
plt.plot(dates_test.values, y_test_real.flatten(),
         label='Реальные значения', linewidth=1.5)
plt.plot(dates_test.values, predictions_real.flatten(),
         label='Прогноз LSTM', linewidth=1.5, alpha=0.8)
plt.xlabel("Дата")
plt.ylabel("Курс USD/UZS")
plt.title("Прогноз LSTM курса USD/UZS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/lstm_forecast.png", dpi=300)
plt.show()