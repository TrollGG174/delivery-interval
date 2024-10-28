import random
from datetime import datetime, timedelta, time

# Скорости для каждого уровня пробок (км/ч)
speeds = {
    1: 60,
    2: 54,
    3: 48,
    4: 42,
    5: 36,
    6: 30,
    7: 24,
    8: 18,
    9: 12,
    10: 6
}

# Погрешности для каждого уровня пробок (в процентах)
errors = {
    1: 0.05,
    2: 0.05,
    3: 0.05,
    4: 0.10,
    5: 0.10,
    6: 0.10,
    7: 0.15,
    8: 0.15,
    9: 0.15,
    10: 0.20
}

# Временные интервалы для доставки (часы)
time_intervals = [
    (time(10, 0), time(12, 0)),
    (time(12, 0), time(14, 0)),
    (time(14, 0), time(16, 0)),
    (time(16, 0), time(18, 0)),
    (time(18, 0), time(20, 0)),
    (time(20, 0), time(22, 0))
]

# Расстояние (обучаем до 100км)
max_distance = 1000
distances_meters = list(range(1, max_distance + 1))

# Функция для генерации времени начала заказа
def generate_order_time():
    start_time = time(6, 0)
    end_time = time(23, 59)
    start_seconds = start_time.hour * 3600 + start_time.minute * 60
    end_seconds = end_time.hour * 3600 + end_time.minute * 60
    random_seconds = random.randint(start_seconds, end_seconds)
    return (datetime.min + timedelta(seconds=random_seconds)).time()

# Функция для определения ближайшего временного интервала
def find_delivery_interval(order_time, delivery_time_minutes):
    order_datetime = datetime.combine(datetime.min, order_time)
    time_in_hours = (order_datetime.hour * 60 + order_datetime.minute + delivery_time_minutes) / 60
    delivery_datetime = order_datetime + timedelta(minutes=delivery_time_minutes)
    delivery_time = delivery_datetime.time()
    if order_time.hour > 22 or time_in_hours > 22:
        return None, None, None
    if delivery_time < time(10, 0):
        return 0, time(10, 0), time(12, 0)
    for idx, (interval_start, interval_end) in enumerate(time_intervals):
        if interval_start <= delivery_time <= interval_end:
            return idx, interval_start, interval_end

    return None, None, None

# Рассчитываем время (в минутах) для проезда каждого расстояния для каждого уровня пробок
def get_union_results():
    union_results = []

    for distance in distances_meters:
        distance = distance * 100  #  берем по 100 метров за раз
        distance_km = distance / 1000  # Переводим в километры
        for level, speed in speeds.items():
            base_time = (distance_km / speed) * 60  # Время в минутах
            error = base_time * errors[level]
            time_with_error = base_time + random.uniform(-error, error)
            order_time = generate_order_time()
            interval, start, end = find_delivery_interval(order_time, time_with_error)
            union_results.append([
                distance,
                level,
                order_time,
                round(time_with_error, 2),
                interval,
                start,
                end
            ])

    return union_results

# print(get_union_results())