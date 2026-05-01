import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузка данных
df = pd.read_csv("data/usd_uzs_daily.csv")

# 2. Дата
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 3. Отсечение до 2017
df = df[df['date'] >= '2017-09-05']

# 4. Проверки
print("Пропуски:\n", df.isna().sum())
print("Дубликаты:", df.duplicated().sum())

# 5. Лаги
for lag in range(1, 8):
    df[f'lag_{lag}'] = df['usd_uzs'].shift(lag)

# 6. Удаляем NaN
df = df.dropna()

# 7. Разделение
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# 8. Нормализация
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train[['usd_uzs']])
test_scaled = scaler.transform(test[['usd_uzs']])

print("Train size:", len(train))
print("Test size:", len(test))
df.to_csv("data/processed_data.csv", index=False)