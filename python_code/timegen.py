import random
from datetime import datetime, timedelta, time

# Скорости для каждого уровня пробок (км/ч)
speeds = {
    1: 55,
    2: 50,
    3: 45,
    4: 40,
    5: 35,
    6: 30,
    7: 24,
    8: 18,
    9: 12,
    10: 6
}

# Погрешности для каждого уровня пробок (в процентах)
errors = {
    1: 0.2,
    2: 0.2,
    3: 0.2,
    4: 0.3,
    5: 0.3,
    6: 0.3,
    7: 0.4,
    8: 0.4,
    9: 0.4,
    10: 0.5
}

# Временные интервалы для доставки (с X часов по Y часов)
time_intervals = [
    (time(10, 0), time(12, 0)),
    (time(12, 0), time(14, 0)),
    (time(14, 0), time(16, 0)),
    (time(16, 0), time(18, 0)),
    (time(18, 0), time(20, 0)),
    (time(20, 0), time(22, 0))
]

# Расстояние (max_distance раз по 100 метров, максимум max_distance * 100 метров)
max_distance = 1000
distances_meters = list(range(1, max_distance + 1))

# Функция для генерации времени начала заказа
def generate_order_time():
    """
    Генерирует случайное время для начала заказа.
    Время будет находиться в диапазоне от 6:00 до 23:59.
    """

    start_time = time(6, 0) # Нижняя граница времени заказа
    end_time = time(23, 59) # Верхняя граница времени заказа

    # Переводим время в секунды для удобного выбора случайного времени
    start_seconds = start_time.hour * 3600 + start_time.minute * 60
    end_seconds = end_time.hour * 3600 + end_time.minute * 60

    # Выбираем случайное количество секунд между началом и концом
    random_seconds = random.randint(start_seconds, end_seconds)

    # Преобразуем обратно в формат времени
    return (datetime.min + timedelta(seconds=random_seconds)).time()

# Функция для определения ближайшего временного интервала
def find_delivery_interval(order_time, delivery_time_minutes):
    """
    Находит ближайший временной интервал доставки для заказа.
    Если доставка не попадает в рабочие интервалы (10:00 - 22:00), возвращает None.

    :param order_time: Время оформления заказа (объект time)
    :param delivery_time_minutes: Время доставки в минутах
    :return: Индекс временного интервала, время начала и конца интервала
    """

    # Конвертируем время заказа в datetime для удобных расчетов
    order_datetime = datetime.combine(datetime.min, order_time)

    # Рассчитываем время доставки в часах
    time_in_hours = (order_datetime.hour * 60 + order_datetime.minute + delivery_time_minutes) / 60

    # Вычисляем фактическое время доставки
    delivery_datetime = order_datetime + timedelta(minutes=delivery_time_minutes)
    delivery_time = delivery_datetime.time()

    # Проверяем, попадает ли доставка за пределы рабочего времени (после 22:00)
    if order_time.hour > 22 or time_in_hours > 22:
        return None, None, None

    # Если доставка происходит до начала интервалов (до 10:00), возвращаем первый интервал
    if delivery_time < time(10, 0):
        return 0, time(10, 0), time(12, 0)

    # Ищем, в какой из временных интервалов попадает доставка
    for idx, (interval_start, interval_end) in enumerate(time_intervals):
        if interval_start <= delivery_time <= interval_end:
            return idx, interval_start, interval_end

    # Если доставка не попала ни в один из интервалов
    return None, None, None

# Рассчитываем время (в минутах) для проезда каждого расстояния для каждого уровня пробок
def get_union_results():
    """
    Генерирует результаты для всех комбинаций расстояний и уровней пробок.
    Возвращает список с данными о расстоянии, уровне пробок, времени заказа,
    расчетном времени доставки и временном интервале.
    """
    union_results = [] # Список для хранения результатов

    # Перебираем расстояния от 100 метров до максимального (1000 * 100 = 100 км)
    for distance in distances_meters:
        distance = distance * 100 # Преобразуем в метры (по 100 метров за шаг)
        distance_km = distance / 1000  # Переводим в километры

        # Для каждого расстояния рассчитываем данные для всех уровней пробок
        for level, speed in speeds.items():
            # Базовое время в минутах: расстояние / скорость
            base_time = (distance_km / speed) * 60

            # Добавляем случайную погрешность (ошибку) в расчет времени
            error = base_time * errors[level]
            time_with_error = base_time + random.uniform(0, error)

            # Генерируем случайное время заказа
            order_time = generate_order_time()

            # Находим временной интервал доставки
            interval, start, end = find_delivery_interval(order_time, time_with_error)

            # Добавляем данные в результирующий список
            union_results.append([
                distance, # Расстояние в метрах
                level, # Уровень пробок
                order_time, # Время оформления заказа
                round(time_with_error, 2), # Время доставки с учетом погрешности
                interval, # Индекс временного интервала
                start, # Время начала интервала
                end # Время конца интервала
            ])

    return union_results

# Пример вызова функции для проверки
# print(get_union_results())