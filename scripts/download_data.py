import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import prepare_dataset
from config import config

if __name__ == "__main__":
    print("Начинаем загрузку данных...")
    train_df, test_df, mapping = prepare_dataset(config)
    print(f"\nГотово! Загружено {len(train_df) + len(test_df)} изображений")