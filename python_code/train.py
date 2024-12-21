import torch
import torch.optim as optim
from models import IntervalAndTimePredictor
from data import get_num_intervals, get_num_categories, get_interval_labels, get_delivery_time, get_preprocessed_data

# Проверяем доступные устройства и выбираем лучшее
if torch.backends.mps.is_available():
    # Если доступна поддержка Metal Performance Shaders (MPS), используем её (для Apple Silicon)
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    # Если доступна видеокарта с CUDA, используем её для ускорения вычислений
    device = torch.device("cuda")
    print("Using CUDA")
else:
    # Если нет доступных ускорителей, используем CPU
    device = torch.device("cpu")
    print("Using CPU")

# Получаем параметры, такие как количество интервалов и категорий, а также закодированные метки
num_intervals = get_num_intervals()
num_categories = get_num_categories()
interval_labels_encoded = get_interval_labels()
delivery_times_categorized = get_delivery_time()

# Функция для тренировки модели
def train_model(num_epochs, split_ratio):
    # split_ratio = 0.8 — это пропорция для разделения данных на тренировочные и тестовые
    X = get_preprocessed_data()
    split_index = int(len(X) * split_ratio)

    # Разделяем данные на тренировочные и тестовые
    X_train, X_test = X[:split_index], X[split_index:]
    y_train_intervals, y_test_intervals = interval_labels_encoded[:split_index], interval_labels_encoded[split_index:]
    y_train_delivery, y_test_delivery = delivery_times_categorized[:split_index], delivery_times_categorized[split_index:]

    # Преобразуем данные в тензоры и отправляем их на выбранное устройство (GPU/CPU)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_intervals = torch.tensor(y_train_intervals, dtype=torch.long).to(device)
    y_test_intervals = torch.tensor(y_test_intervals, dtype=torch.long).to(device)
    y_train_delivery = torch.tensor(y_train_delivery, dtype=torch.long).to(device)
    y_test_delivery = torch.tensor(y_test_delivery, dtype=torch.long).to(device)

    # Инициализация модели с количеством интервалов и категорий
    model = IntervalAndTimePredictor(input_size=3, num_intervals=num_intervals, num_categories=num_categories).to(device)

    # Определение функций потерь и оптимизатора
    criterion_intervals = torch.nn.CrossEntropyLoss()  # Для категориальной классификации интервалов
    criterion_delivery_time = torch.nn.CrossEntropyLoss()  # Для категориальной классификации времени доставки
    optimizer = optim.RAdam(model.parameters())  # Используем адаптивный оптимизатор RAdam

    # Обучение модели
    for epoch in range(num_epochs):
        model.train()  # Устанавливаем модель в режим тренировки
        optimizer.zero_grad()  # Обнуляем градиенты
        interval_outputs, delivery_time_outputs = model(X_train)  # Прогоняем данные через модель
        loss_intervals = criterion_intervals(interval_outputs, y_train_intervals)  # Потери для интервалов
        loss_delivery_time = criterion_delivery_time(delivery_time_outputs, y_train_delivery)  # Потери для времени доставки
        loss = loss_intervals + loss_delivery_time  # Общая потеря
        loss.backward()  # Вычисляем градиенты
        optimizer.step()  # Обновляем веса модели

        # Выводим информацию о прогрессе обучения каждые 100 эпох
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():  # Отключаем градиенты для вычислений без их использования
                interval_predictions = torch.argmax(interval_outputs, dim=1)  # Прогнозируем интервалы
                delivery_time_predictions = torch.argmax(delivery_time_outputs, dim=1)  # Прогнозируем время доставки
                interval_accuracy = (interval_predictions == y_train_intervals).sum().item() / len(y_train_intervals)  # Точность по интервалам
                delivery_time_accuracy = (delivery_time_predictions == y_train_delivery).sum().item() / len(y_train_delivery)  # Точность по времени доставки

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Interval Accuracy: {interval_accuracy:.4f}, Delivery Time Accuracy: {delivery_time_accuracy:.4f}")

    # Тестирование модели
    model.eval()  # Переводим модель в режим оценки
    with torch.no_grad():
        interval_outputs_test, delivery_time_outputs_test = model(X_test)  # Прогоняем тестовые данные
        interval_test_predictions = torch.argmax(interval_outputs_test, dim=1)  # Прогнозируем интервалы на тестовых данных
        delivery_time_test_predictions = torch.argmax(delivery_time_outputs_test, dim=1)  # Прогнозируем время доставки на тестовых данных
        interval_test_accuracy = (interval_test_predictions == y_test_intervals).sum().item() / len(y_test_intervals)  # Точность по интервалам на тесте
        delivery_time_test_accuracy = (delivery_time_test_predictions == y_test_delivery).sum().item() / len(y_test_delivery)  # Точность по времени доставки на тесте

    print(f"Test Interval Accuracy: {interval_test_accuracy:.4f}, Test Delivery Time Accuracy: {delivery_time_test_accuracy:.4f}")

    # Сохранение модели
    model_path = "interval_and_time_predictor.pth"  # Путь для сохранения модели
    torch.save(model.state_dict(), model_path)  # Сохраняем веса модели в файл