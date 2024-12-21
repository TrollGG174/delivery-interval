import numpy as np

# Класс Config используется для хранения и управления глобальными переменными настройки
class Config:
    def __init__(self):
        # Инициализация максимального расстояния, уровня трафика и максимального количества часов в сутках
        self.max_distance = 0.0
        self.max_traffic = 0.0
        self.max_order_hours = 24.0  # Максимальное количество часов в сутках

    # Метод для установки максимального расстояния
    def set_max_distance(self, value):
        self.max_distance = value

    # Метод для установки максимального уровня трафика
    def set_max_traffic(self, value):
        self.max_traffic = value

    # Метод для получения максимального расстояния
    def get_max_distance(self):
        return self.max_distance

    # Метод для получения максимального уровня трафика
    def get_max_traffic(self):
        return self.max_traffic

    # Метод для получения максимального количества часов в сутках
    def get_max_order_hours(self):
        return self.max_order_hours


# Создание глобального объекта конфигурации
config = Config()

# Глобальные функции для получения значений из объекта конфигурации
def get_max_distance():
    return config.get_max_distance()

def get_max_traffic():
    return config.get_max_traffic()

def get_max_order_hours():
    return config.get_max_order_hours()

# Функция для нормализации данных
def normalize_data(distances, traffic_levels, order_hours):
    # Обновление максимального расстояния и уровня трафика в объекте конфигурации
    config.set_max_distance(np.max(distances))
    config.set_max_traffic(np.max(traffic_levels))
    max_order_hours = 24.0  # Максимальное количество часов в сутках (локальная переменная)

    # Нормализация каждого массива данных
    distances /= config.get_max_distance()
    traffic_levels /= config.get_max_traffic()
    order_hours /= max_order_hours

    return distances, traffic_levels, order_hours

# Функция для категоризации времени доставки
def categorize_delivery_time(time):
    # Возвращает категорию времени на основе заданных интервалов
    if time < 5:
        return 0
    elif time < 15:
        return 1
    elif time < 30:
        return 2
    elif time < 60:
        return 3
    elif time < 120:
        return 4
    elif time < 240:
        return 5
    elif time < 480:
        return 6
    else:
        return 7

# Функция для предварительной обработки данных
def preprocess_data(data):
    # Инициализация массивов для хранения различных данных
    distances = []
    traffic_levels = []
    order_times = []
    delivery_times = []
    interval_labels = []

    # Разбивка входных данных на отдельные массивы
    for entry in data:
        distances.append(entry[0])
        traffic_levels.append(entry[1])
        order_times.append(entry[2])
        delivery_times.append(entry[3])
        if entry[4] is not None:
            interval_labels.append(entry[4])
        else:
            interval_labels.append('Empty')  # Если метка интервала отсутствует, устанавливаем значение 'Empty'

    # Преобразование массивов в формат NumPy и преобразование типов данных
    distances = np.array(distances).astype(float)
    traffic_levels = np.array(traffic_levels).astype(float)
    order_times = np.array(order_times)
    delivery_times = np.array(delivery_times).astype(float)

    # Вычисление времени заказа в часах (например, 10:30 -> 10.5)
    order_hours = []
    for time_obj in order_times:
        hours = time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600
        order_hours.append(hours)
    order_hours = np.array(order_hours)

    # Нормализация данных
    distances, traffic_levels, order_hours = normalize_data(distances, traffic_levels, order_hours)

    # Возвращение подготовленных данных
    return distances, traffic_levels, order_hours, delivery_times, interval_labels