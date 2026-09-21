import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from tqdm import tqdm
from datetime import datetime
import json

class BirdClassifier(nn.Module):
    """Классификатор птиц (без предобученных весов)"""
    
    def __init__(self, num_classes=5, model_name='simple_cnn'):
        super(BirdClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'efficientnet_b0':
            self.backbone = self._create_simple_efficientnet(num_classes, width_mult=1.0, depth_mult=1.0)
        elif model_name == 'efficientnet_b3':
            self.backbone = self._create_simple_efficientnet(num_classes, width_mult=1.2, depth_mult=1.4)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'simple_cnn':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(f"Неподдерживаемая модель: {model_name}")
    
    def _create_simple_efficientnet(self, num_classes, width_mult=1.0, depth_mult=1.0):
        def make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        base_config = [
            [32, 16, 3, 1, 1, 1],
            [16, 24, 3, 2, 6, 2],
            [24, 40, 5, 2, 6, 2],
            [40, 80, 3, 2, 6, 3],
            [80, 112, 5, 1, 6, 3],
            [112, 192, 5, 2, 6, 4],
            [192, 320, 3, 1, 6, 1]
        ]
        
        config = []
        for in_c, out_c, kernel, stride, exp_ratio, repeats in base_config:
            out_c = make_divisible(out_c * width_mult)
            repeats = int(math.ceil(repeats * depth_mult))
            config.append([in_c, out_c, kernel, stride, exp_ratio, repeats])
        
        class MBConv(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
                super().__init__()
                self.use_residual = stride == 1 and in_channels == out_channels
                hidden_dim = in_channels * expand_ratio
                layers = []
                if expand_ratio != 1:
                    layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
                    layers.append(nn.BatchNorm2d(hidden_dim))
                    layers.append(nn.ReLU6(inplace=True))
                layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                              padding=kernel_size//2, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels)
                ])
                self.conv = nn.Sequential(*layers)
            def forward(self, x):
                if self.use_residual:
                    return x + self.conv(x)
                return self.conv(x)
        
        class SimpleEfficientNet(nn.Module):
            def __init__(self):
                super().__init__()
                first_channels = make_divisible(32 * width_mult)
                self.features = [nn.Sequential(
                    nn.Conv2d(3, first_channels, 3, 2, padding=1, bias=False),
                    nn.BatchNorm2d(first_channels),
                    nn.ReLU6(inplace=True)
                )]
                in_channels = first_channels
                for i, (in_c, out_c, kernel, stride, exp_ratio, repeats) in enumerate(config):
                    out_channels = make_divisible(out_c)
                    for j in range(repeats):
                        stride = stride if j == 0 else 1
                        self.features.append(
                            MBConv(in_channels if j == 0 else out_channels,
                                  out_channels, kernel, stride, exp_ratio)
                        )
                        in_channels = out_channels
                self.features = nn.Sequential(*self.features)
                last_channels = make_divisible(1280 * max(1.0, width_mult))
                self.classifier = nn.Sequential(
                    nn.Conv2d(in_channels, last_channels, 1, bias=False),
                    nn.BatchNorm2d(last_channels),
                    nn.ReLU6(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(last_channels, num_classes)
                )
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        return SimpleEfficientNet()
    
    def forward(self, x):
        return self.backbone(x)

class Trainer:
    def __init__(self, config, device=None):
        self.config = config
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Используется устройство: {self.device}")
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc="Обучение")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100.*correct/total})
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate(self, model, test_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Валидация")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100.*correct/total})
        return running_loss / len(test_loader), 100. * correct / total
    
    def train(self, model, train_loader, test_loader, model_name="bird_classifier"):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 2)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        print(f"\nНачинаем обучение {model.model_name} с нуля...")
        print(f"Эпох: {self.config.num_epochs}, LR: {self.config.learning_rate * 2}")
        for epoch in range(self.config.num_epochs):
            print(f"\nЭпоха {epoch+1}/{self.config.num_epochs}")
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(model, test_loader, criterion)
            scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config.__dict__,
                    'model_name': model.model_name
                }, self.config.model_save_path)
                print(f"Сохранена лучшая модель с точностью {val_acc:.2f}%")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"\nОбучение завершено! Лучшая точность: {best_acc:.2f}%")
        return history, best_acc

def run_experiment(config, model_name='simple_cnn'):
    import pandas as pd
    from src.data_processing import create_data_loaders
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    train_loader, test_loader = create_data_loaders(train_df, test_df, config)
    model = BirdClassifier(num_classes=config.num_classes, model_name=model_name)
    print(f"Создана модель {model_name}")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(config)
    history, best_acc = trainer.train(model, train_loader, test_loader, model_name)
    metrics = {
        'model': model_name,
        'best_val_accuracy': best_acc,
        'final_train_accuracy': history['train_acc'][-1],
        'final_val_accuracy': history['val_acc'][-1],
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }
    metrics_file = "data/processed/metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics