from preprocessing import preprocess_data, categorize_delivery_time
import timegen
import numpy as np

# Загружаем данные о доставке, включая расстояния, уровень трафика, время заказа и метки интервалов
data = timegen.get_union_results()

# Обрабатываем сырые данные: разбиваем их на расстояния, уровни трафика, часы заказов, время доставки и метки интервалов
distances, traffic_levels, order_hours, delivery_times, interval_labels = preprocess_data(data)

# Преобразование категориальных меток интервалов в числовые индексы для удобства работы с машинным обучением
interval_labels = [str(label) for label in interval_labels]  # Преобразуем метки в строки
category_to_index = {category: idx for idx, category in enumerate(sorted(set(interval_labels)))}  # Создаем отображение категория -> индекс
interval_labels_encoded = np.array([category_to_index[label] for label in interval_labels])  # Кодируем категории в числовой массив

# Категоризация времени доставки
delivery_times_categorized = np.array([categorize_delivery_time(t) for t in delivery_times])

# Улучшенное разделение данных: формируем входные данные X из расстояний, уровня трафика и часов заказов
X = np.vstack((distances, traffic_levels, order_hours)).T  # Транспонируем массив, чтобы получить набор признаков в строках
indices = np.arange(len(X))  # Генерируем индексы для всех данных
np.random.shuffle(indices)  # Перемешиваем индексы для случайного порядка данных
X = X[indices]  # Перемешиваем сами данные
interval_labels_encoded = interval_labels_encoded[indices]  # Перемешиваем метки интервалов доставки
delivery_times_categorized = delivery_times_categorized[indices]  # Перемешиваем категории времени доставки

# Инициализация параметров модели
num_intervals = len(category_to_index)  # Количество уникальных интервалов доставки
num_categories = len(set(delivery_times_categorized))  # Количество уникальных категорий времени доставки

# Функция для получения количества интервалов доставки
def get_num_intervals():
    return num_intervals

# Функция для получения количества категорий времени доставки
def get_num_categories():
    return num_categories

# Функция для получения категорий времени доставки
def get_delivery_time():
    return delivery_times_categorized

# Функция для получения кодированных меток интервалов доставки
def get_interval_labels():
    return interval_labels_encoded

# Функция для получения обработанных данных
def get_preprocessed_data():
    return X