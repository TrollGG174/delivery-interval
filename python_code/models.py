import torch
import torch.nn as nn

# Определяем класс нейронной сети для предсказания интервалов доставки и времени доставки
class IntervalAndTimePredictor(nn.Module):
    def __init__(self, input_size, num_intervals, num_categories):
        """
        Конструктор класса.
        Аргументы:
        - input_size: размер входных данных (количество признаков).
        - num_intervals: количество возможных интервалов доставки (классы для первой задачи).
        - num_categories: количество категорий времени доставки (классы для второй задачи).
        """
        super(IntervalAndTimePredictor, self).__init__()
        # Определяем архитектуру сети:
        # Входной слой -> скрытые слои -> выходные слои для двух задач
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        # Два отдельных выходных слоя для задач классификации:
        self.interval_output = nn.Linear(64, num_intervals)
        self.delivery_time_output = nn.Linear(64, num_categories)
        # Dropout для регуляризации, чтобы избежать переобучения
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Определяет прямой проход (forward pass) через сеть.
        Аргументы:
        - x: входные данные (батч признаков).

        Возвращает:
        - interval_output: логиты для задачи предсказания интервалов доставки.
        - delivery_time_output: логиты для задачи классификации времени доставки.
        """
        # Последовательное прохождение данных через скрытые слои с активацией tanh и регуляризацией
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc4(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc5(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc6(x))
        # Два отдельных выхода для различных задач
        interval_output = self.interval_output(x)
        delivery_time_output = self.delivery_time_output(x)
        # Возвращаем оба выхода
        return interval_output, delivery_time_output