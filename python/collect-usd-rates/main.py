import requests
import csv
import time
from datetime import date, timedelta

START_DATE = date(2015, 1, 1)
END_DATE   = date(2025, 12, 31)
OUTPUT_FILE = "data/usd_uzs_daily.csv"

def fetch_rate(d: date):
    url = f"https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/{d.isoformat()}/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            return data[0]["Rate"]   # строка вида "12345.67"
    except Exception as e:
        print(f"  Ошибка {d}: {e}")
    return None

def main():
    import os
    os.makedirs("data", exist_ok=True)

    current = START_DATE
    rows = []
    prev_rate = None

    while current <= END_DATE:
        rate = fetch_rate(current)

        if rate is None:
            # Выходные/праздники — ЦБ не публикует курс, берём предыдущий
            rate = prev_rate
        else:
            prev_rate = rate

        if rate is not None:
            rows.append({"date": current.isoformat(), "usd_uzs": rate})

        print(f"{current}  →  {rate}")
        current += timedelta(days=1)
        time.sleep(0.1)   # 100 мс пауза, чтобы не перегружать сервер

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "usd_uzs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово! Сохранено {len(rows)} строк в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()