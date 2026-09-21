import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Пути к данным
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    images_dir: str = "data/images"
    
    # Названия видов (из CSV файлов)
    species_files: List[str] = None
    
    # Параметры модели
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 0.001
    num_classes: int = 5
    
    # Параметры обучения
    use_augmentation: bool = True
    
    # Пути для сохранения
    train_csv: str = "data/processed/train.csv"
    test_csv: str = "data/processed/test.csv"
    species_mapping: str = "data/processed/species_mapping.json"
    model_save_path: str = "models/best_model.pth"
    
    # Имя эксперимента
    experiment_name: str = "bird_classification"
    
    def __post_init__(self):
        if self.species_files is None:
            self.species_files = [
                "Anas_platyrhynchos.csv",
                "Coloeus_monedula.csv",
                "Columba_livia_domestica.csv",
                "Corvus_cornix.csv",
                "Larus_argentatus.csv"
            ]
        self.species_files = [os.path.join(self.raw_data_dir, f) for f in self.species_files]

config = Config()