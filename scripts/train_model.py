import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_training import run_experiment
from config import config

if __name__ == "__main__":
    print("Начинаем обучение модели с нуля...")
    config.num_epochs = 20
    metrics = run_experiment(config, model_name='simple_cnn')
    print(f"\nГотово! Лучшая точность: {metrics['best_val_accuracy']:.2f}%")