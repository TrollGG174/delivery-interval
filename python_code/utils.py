import torch
import numpy as np
from datetime import datetime
from models import IntervalAndTimePredictor
from data import get_num_intervals, get_num_categories
from preprocessing import get_max_distance, get_max_traffic, get_max_order_hours


# Функция для предсказания интервала доставки и времени доставки
def predict_interval_and_time(distance, traffic_level, order_time, model):
    # Получаем максимальные значения для нормализации входных данных
    max_distance = get_max_distance()  # Максимальное расстояние
    max_traffic = get_max_traffic()  # Максимальный уровень трафика
    max_order_hours = get_max_order_hours()  # Максимальное время заказа

    # Преобразуем время оформления заказа в количество часов с начала дня
    order_hour = order_time.hour + order_time.minute / 60 + order_time.second / 3600

    # Нормализуем входные данные вручную, делим их на максимальные значения
    input_data = np.array([[distance / max_distance, traffic_level / max_traffic, order_hour / max_order_hours]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Преобразуем данные в тензор для подачи в модель

    # Отключаем расчет градиентов для предотвращения лишней нагрузки на память
    with torch.no_grad():
        # Получаем предсказания от модели
        interval_output, delivery_time_output = model(input_tensor)

        # Получаем индекс самого вероятного интервала
        predicted_interval_idx = torch.argmax(interval_output).item()
        # Определяем, какой интервал соответствует предсказанному индексу
        time_intervals = [(10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), ('0')]
        predicted_interval = time_intervals[predicted_interval_idx]

        # Получаем индекс самого вероятного времени доставки
        predicted_delivery_time_idx = torch.argmax(delivery_time_output).item()
        # Определяем, какое время доставки соответствует предсказанному индексу
        delivery_time_categories = ['меньше 5 минут', 'от 5 до 15 минут', 'от 15 до 30 минут', 'от 30 минут до 1 часа',
                                    'от 1 до 2 часов', 'от 2 до 4 часов', 'от 4 до 8 часов', 'больше 8 часов']
        predicted_delivery_time = delivery_time_categories[predicted_delivery_time_idx]

    # Возвращаем предсказанный интервал и время доставки
    return predicted_interval, predicted_delivery_time


# Упрощённая функция вызова предсказания
def intervals(distance, traffic, time):
    # Преобразуем строку времени в объект времени
    time_object = datetime.strptime(time, "%H:%M").time()
    traffic = float(traffic)  # Преобразуем уровень трафика в число с плавающей точкой

    # Загружаем модель для предсказания
    model_path = "interval_and_time_predictor.pth"
    loaded_model = IntervalAndTimePredictor(3, num_intervals=get_num_intervals(), num_categories=get_num_categories())
    # Загружаем веса обученной модели
    loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # Устанавливаем модель в режим оценки (без обучения)
    loaded_model.eval()

    # Вызываем функцию предсказания и возвращаем результат
    return predict_interval_and_time(distance, traffic, time_object, loaded_model)