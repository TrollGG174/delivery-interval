import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import timegen
import sys
import ready_set


# Пример данных (расстояние в метрах, уровень трафика, время начала заказа (часы, минуты, секунды),
# время доставки в часах, минимальное время интервала, максимальное время интервала, идентификатор интервала)
data = timegen.get_union_results()
# data = ready_set.get_ready_set()

distances = []
traffic_levels = []
order_times = []
interval_labels = []
delivery_times = []

time_intervals = [(10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), ('Empty for today')]
delivery_time_categories = ['<5 minutes', '5-15 minutes', '15-30 minutes', '30-1 hour', '1-2 hour', '2-4 hour', '4-8 hour', '>8 hour']

for entry in data:
    distances.append(entry[0])
    traffic_levels.append(entry[1])
    order_times.append(entry[2])
    delivery_times.append(entry[3])
    if entry[4] is not None:
        interval_labels.append(entry[4])
    else:
        interval_labels.append('Empty')

distances = np.array(distances).astype(float)
traffic_levels = np.array(traffic_levels).astype(float)
order_times = np.array(order_times)
delivery_times = np.array(delivery_times).astype(float)

# Преобразование времени оформления заказа в количество часов с начала дня
order_hours = []
for time_obj in order_times:
    hours = time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600
    order_hours.append(hours)

order_hours = np.array(order_hours)

# Нормализация данных с использованием MinMaxScaler
scaler = MinMaxScaler()
X = np.vstack((distances, traffic_levels, order_hours)).T
X_scaled = scaler.fit_transform(X)

# Преобразование данных в тензоры
X_train, X_test, y_train_intervals, y_test_intervals, y_train_delivery, y_test_delivery = train_test_split(
    X_scaled, interval_labels, delivery_times, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Преобразование меток в тензоры и обработка отсутствующих интервалов
y_train_intervals = np.array(y_train_intervals)
valid_train_indices = y_train_intervals != -1
X_train = X_train[valid_train_indices]
y_train_intervals = y_train_intervals[valid_train_indices]

y_test_intervals = np.array(y_test_intervals)
valid_test_indices = y_test_intervals != -1
X_test = X_test[valid_test_indices]
y_test_intervals = y_test_intervals[valid_test_indices]

label_encoder = LabelEncoder()
y_train_intervals = label_encoder.fit_transform(y_train_intervals)
y_train_intervals = torch.tensor(y_train_intervals, dtype=torch.long)
y_test_intervals = label_encoder.transform(y_test_intervals)
y_test_intervals = torch.tensor(y_test_intervals, dtype=torch.long)

# Категоризация времени доставки
def categorize_delivery_time(time):
    if time < 5:
        return 0  # Меньше 5 минут
    elif time < 15:
        return 1  # От 5 до 15 минут
    elif time < 30:
        return 2  # От 15 минут до 30 минут
    elif time < 60:
        return 3  # От 30 минут до часа
    elif time < 120:
        return 4  # От часа до 2 часов
    elif time < 240:
        return 5  # От 2 часов до 4 часов
    elif time < 480:
        return 6  # От 4 часов до 8 часов
    else:
        return 7  # Больше 8 часов

y_train_delivery = np.array([categorize_delivery_time(t) for t in y_train_delivery])
y_test_delivery = np.array([categorize_delivery_time(t) for t in y_test_delivery])
y_train_delivery = torch.tensor(y_train_delivery, dtype=torch.long)
y_test_delivery = torch.tensor(y_test_delivery, dtype=torch.long)

# Определение модели
class IntervalAndTimePredictor(nn.Module):
    def __init__(self):
        super(IntervalAndTimePredictor, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.interval_output = nn.Linear(64, len(np.unique(y_train_intervals)))
        self.delivery_time_output = nn.Linear(64, 8)  # 8 категорий для времени доставки
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
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
        interval_output = self.interval_output(x)
        delivery_time_output = self.delivery_time_output(x)
        return interval_output, delivery_time_output

model = IntervalAndTimePredictor()
criterion_intervals = nn.CrossEntropyLoss()
criterion_delivery_time = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

# Обучение модели
num_epochs = 10000

# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     interval_outputs, delivery_time_outputs = model(X_train)
#     loss_intervals = criterion_intervals(interval_outputs, y_train_intervals)
#     loss_delivery_time = criterion_delivery_time(delivery_time_outputs, y_train_delivery)
#     loss = loss_intervals + loss_delivery_time
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#
#     # Вычисление точности (accuracy) для интервалов и времени доставки
#     with torch.no_grad():
#         interval_predictions = torch.argmax(interval_outputs, dim=1)
#         delivery_time_predictions = torch.argmax(delivery_time_outputs, dim=1)
#         interval_accuracy = (interval_predictions == y_train_intervals).sum().item() / len(y_train_intervals)
#         delivery_time_accuracy = (delivery_time_predictions == y_train_delivery).sum().item() / len(y_train_delivery)
#
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
#               f'Interval Accuracy: {interval_accuracy:.4f}, Delivery Time Accuracy: {delivery_time_accuracy:.4f}')

# Проверка модели на тестовых данных
model.eval()
with torch.no_grad():
    interval_outputs_test, delivery_time_outputs_test = model(X_test)
    interval_test_predictions = torch.argmax(interval_outputs_test, dim=1)
    delivery_time_test_predictions = torch.argmax(delivery_time_outputs_test, dim=1)
    interval_test_accuracy = (interval_test_predictions == y_test_intervals).sum().item() / len(y_test_intervals)
    delivery_time_test_accuracy = (delivery_time_test_predictions == y_test_delivery).sum().item() / len(y_test_delivery)

# print(f'Test Interval Accuracy: {interval_test_accuracy:.4f}, Test Delivery Time Accuracy: {delivery_time_test_accuracy:.4f}')

# Сохранение модели
model_path = "interval_and_time_predictor.pth"
# torch.save(model.state_dict(), model_path)

# Загрузка модели
loaded_model = IntervalAndTimePredictor()
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

# Тестирование модели на новых данных
def predict_interval_and_time(distance, traffic_level, order_time, model, scaler, label_encoder):
    # Преобразование времени оформления заказа в количество часов с начала дня
    order_hour = order_time.hour + order_time.minute / 60 + order_time.second / 3600

    # Нормализация входных данных
    input_data = np.array([[distance, traffic_level, order_hour]])
    input_data_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        interval_output, delivery_time_output = model(input_tensor)
        predicted_interval_idx = torch.argmax(interval_output).item()
        # print(predicted_interval_idx)
        predicted_interval = time_intervals[predicted_interval_idx]
        predicted_delivery_time_idx = torch.argmax(delivery_time_output).item()
        # print(predicted_delivery_time_idx)
        predicted_delivery_time = delivery_time_categories[predicted_delivery_time_idx]

    return predicted_interval, predicted_delivery_time

def intervals(distance, traffic, time):
    time_object = datetime.strptime(time, "%H:%M").time()
    predicted_interval, predicted_delivery_time = predict_interval_and_time(distance, traffic, time_object, loaded_model, scaler, label_encoder)
    return predicted_interval, predicted_delivery_time

# if __name__ == "__main__":
    # Extract arguments from the command line
    # distance = sys.argv[1]
    # traffic = sys.argv[2]
    #
    # # Call the function and print the result
    # predicted_interval, predicted_delivery_time = predict_interval_and_time(distance, traffic, time(8, 40, 34), loaded_model, scaler, label_encoder)
    # print(predicted_interval)
    # print(predicted_delivery_time)

# Пример предсказания с использованием загруженной модели
# predicted_interval, predicted_delivery_time = predict_interval_and_time(5000, 10, time(19, 59, 34), loaded_model, scaler, label_encoder)
# print(f'Predicted delivery interval: {predicted_interval}, Predicted delivery time: {predicted_delivery_time}')
# predicted_interval, predicted_delivery_time = predict_interval_and_time(123, 3, time(13, 59, 34), loaded_model, scaler, label_encoder)
# print(f'Predicted delivery interval: {predicted_interval}, Predicted delivery time: {predicted_delivery_time}')
# predicted_interval, predicted_delivery_time = predict_interval_and_time(50000, 10, time(16, 59, 34), loaded_model, scaler, label_encoder)
# print(f'Predicted delivery interval: {predicted_interval}, Predicted delivery time: {predicted_delivery_time}')