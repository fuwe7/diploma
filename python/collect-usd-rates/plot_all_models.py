import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Загрузка данных
arima_roll = pd.read_csv('results/arima_forecast.csv', parse_dates=['date'])
arima_stat = pd.read_csv('results/arima_static_forecast.csv', parse_dates=['date'])
lstm = pd.read_csv('results/lstm_forecast.csv', parse_dates=['date'])
rf = pd.read_csv('results/rf_forecast.csv', parse_dates=['date'])

# График
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(arima_roll['date'], arima_roll['actual'], color='black', linewidth=1.5, label='Реальный курс', zorder=5)
ax.plot(arima_roll['date'], arima_roll['forecast'], color='#2196F3', linewidth=1.0, alpha=0.85, label='ARIMA (rolling), MAPE=0.15%')
ax.plot(arima_stat['date'], arima_stat['forecast'], color='#FF9800', linewidth=1.0, alpha=0.85, label='ARIMA (static), MAPE=1.89%')
ax.plot(lstm['date'], lstm['forecast'], color='#4CAF50', linewidth=1.0, alpha=0.85, label='LSTM, MAPE=0.33%')
ax.plot(rf['date'], rf['forecast'], color='#F44336', linewidth=1.0, alpha=0.85, label='Random Forest, MAPE=0.88%')

ax.set_xlabel('Дата', fontsize=12)
ax.set_ylabel('Курс USD/UZS (сум)', fontsize=12)
ax.set_title('Сравнение прогнозов всех моделей на тестовой выборке', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('plots/plot_all_models.png', dpi=200, bbox_inches='tight')
print("Сохранено: plots/plot_all_models.png")
plt.show()