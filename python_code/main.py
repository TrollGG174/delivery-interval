from train import train_model
from dotenv import load_dotenv
import os

# Загрузка переменных окружения из .env файла
load_dotenv()

# Основной скрипт
if __name__ == "__main__":
    # Параметры из .env
    should_train = os.getenv("SHOULD_TRAIN", "True") == "True"
    num_epochs = int(os.getenv("NUM_EPOCHS", 1000))
    train_test_split_ratio = float(os.getenv("TRAIN_TEST_SPLIT_RATIO", 0.8))

    if should_train:
        train_model(num_epochs=num_epochs, split_ratio=train_test_split_ratio)
    else:
        print("Обучение модели пропущено.")