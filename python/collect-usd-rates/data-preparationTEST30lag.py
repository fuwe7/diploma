import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузка данных
df = pd.read_csv("data/usd_uzs_daily.csv")

# 2. Дата
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 3. Отсечение — с даты либерализации
df = df[df['date'] >= '2017-09-05']

# 4. Проверки
print("Пропуски:\n", df.isna().sum())
print("Дубликаты:", df.duplicated().sum())

# 5. Разделение СНАЧАЛА — до нормализации
train_size = int(len(df) * 0.8)
train = df[:train_size].copy()
test = df[train_size:].copy()

# 6. Нормализация
scaler = MinMaxScaler()
train['usd_uzs_scaled'] = scaler.fit_transform(train[['usd_uzs']])
test['usd_uzs_scaled'] = scaler.transform(test[['usd_uzs']])

# 7. Лаги — теперь из нормализованной колонки
for lag in range(1, 31):
    train[f'lag_{lag}'] = train['usd_uzs_scaled'].shift(lag)
    test[f'lag_{lag}'] = test['usd_uzs_scaled'].shift(lag)

# 8. Удаляем NaN
train = train.dropna()
test = test.dropna()

# 9. Сохранение
full = pd.concat([train, test])
full.to_csv("data/processed_dataTEST30lag.csv", index=False)

print("Train size:", len(train))
print("Test size:", len(test))
print(full.head(3).to_string())