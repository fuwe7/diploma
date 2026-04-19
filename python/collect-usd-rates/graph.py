import pandas as pd
import matplotlib.pyplot as plt

# Загружаем CSV
df = pd.read_csv("data/usd_uzs_daily.csv")   # ← убедись, что путь правильный

# Приводим дату к формату datetime
df['date'] = pd.to_datetime(df['date'])

# Сортируем по дате
df = df.sort_values('date')

# === ИСПРАВЛЕНИЕ ЗДЕСЬ ===

plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['usd_uzs'], linewidth=2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Стиль (лучше ставить в самом начале)
plt.style.use('ggplot')


plt.xlabel("Дата", fontsize=14)
plt.ylabel("Курс (сумов за 1 USD)", fontsize=14)

# Увеличиваем шрифты осей
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Вертикальная линия — момент резкой деноминации в 2017 году
plt.axvline(pd.to_datetime('2017-09-05'), color='red', linestyle='--', linewidth=1.5, label='Либерализация валютного рынка (2017)')
plt.legend(fontsize=12)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Сохраняем в высоком качестве
plt.savefig("usd_uzs_plot.png", dpi=300, bbox_inches='tight')

plt.show()