import pandas as pd
import requests
import os
from tqdm import tqdm
import time
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def download_images_from_csv(csv_path, species_name, output_dir="data/images"):
    """
    Скачивает изображения из CSV файла
    """
    df = pd.read_csv(csv_path)
    print(f"Загрузка изображений для {species_name}: {len(df)} записей")
    
    species_dir = Path(output_dir) / species_name
    species_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_images = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Скачивание {species_name}"):
        try:
            img_url = row['image_url']
            if pd.isna(img_url):
                continue
                
            filename = f"{species_name}_{row['id']}.jpg"
            filepath = species_dir / filename
            
            if filepath.exists():
                downloaded_images.append({
                    'species': species_name,
                    'filepath': str(filepath),
                    'id': row['id'],
                    'scientific_name': row['scientific_name'],
                    'common_name': row['common_name']
                })
                continue
            
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_images.append({
                    'species': species_name,
                    'filepath': str(filepath),
                    'id': row['id'],
                    'scientific_name': row['scientific_name'],
                    'common_name': row['common_name']
                })
                time.sleep(0.1)
        except Exception as e:
            print(f"Ошибка при загрузке {img_url}: {e}")
            continue
    
    print(f"Успешно скачано {len(downloaded_images)}/{len(df)} изображений для {species_name}")
    return pd.DataFrame(downloaded_images)

def prepare_dataset(config):
    """
    Подготавливает полный датасет для обучения
    """
    print("Подготовка датасета...")
    
    Path(config.images_dir).mkdir(parents=True, exist_ok=True)
    Path(config.processed_data_dir).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    all_data = []
    
    for csv_file in config.species_files:
        if os.path.exists(csv_file):
            species_name = Path(csv_file).stem
            df = download_images_from_csv(csv_file, species_name, config.images_dir)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise ValueError("Не удалось загрузить ни одного изображения!")
    
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Всего загружено {len(full_df)} изображений")
    
    unique_species = sorted(full_df['species'].unique())
    species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
    idx_to_species = {idx: species for species, idx in species_to_idx.items()}
    
    full_df['label'] = full_df['species'].map(species_to_idx)
    
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        stratify=full_df['label'],
        random_state=42
    )
    
    mapping_data = {
        'species_to_idx': species_to_idx,
        'idx_to_species': idx_to_species,
        'class_names': {idx: species for idx, species in enumerate(unique_species)}
    }
    
    with open(config.species_mapping, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    train_df.to_csv(config.train_csv, index=False)
    test_df.to_csv(config.test_csv, index=False)
    
    print(f"Тренировочных образцов: {len(train_df)}")
    print(f"Тестовых образцов: {len(test_df)}")
    print(f"Классов: {len(unique_species)}")
    
    return train_df, test_df, mapping_data

def get_transforms(image_size=224, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])

class BirdDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        try:
            image = Image.open(row['filepath']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label

def create_data_loaders(train_df, test_df, config):
    train_transform = get_transforms(config.image_size, augment=True)
    test_transform = get_transforms(config.image_size, augment=False)
    
    train_dataset = BirdDataset(train_df, train_transform)
    test_dataset = BirdDataset(test_df, test_transform)
    
    print(f"Размер тренировочного датасета: {len(train_dataset)}")
    print(f"Размер тестового датасета: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, test_loader