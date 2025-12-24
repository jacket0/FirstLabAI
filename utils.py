import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch  
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class StanfordDogsDataset(Dataset):
    """Кастомный датасет для Stanford Dogs"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(labels)))}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.labels[idx]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size=32, val_size=0.15, test_size=0.15, random_state=42):
    """
    Загрузка данных и создание DataLoader'ов
    Разделение: 70% train, 15% val, 15% test
    """
    # Сбор всех изображений и меток
    image_paths = []
    labels = []
    
    images_dir = os.path.join(data_dir, 'images', 'Images')
    for breed_folder in os.listdir(images_dir):
        breed_path = os.path.join(images_dir, breed_folder)
        if not os.path.isdir(breed_path):
            continue
        
        breed_name = breed_folder.split('-', 1)[1]  # Извлекаем название породы
        
        for img_file in os.listdir(breed_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(breed_path, img_file))
                labels.append(breed_name)
    
    print(f"Всего изображений: {len(image_paths)}")
    print(f"Количество классов: {len(set(labels))}")
    
    # Разделение на train+val и test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Разделение train и val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size/(1-test_size),  # корректируем размер val
        random_state=random_state, 
        stratify=train_val_labels
    )
    
    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создание датасетов
    train_dataset = StanfordDogsDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = StanfordDogsDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = StanfordDogsDataset(test_paths, test_labels, transform=eval_transform)
    
    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

def calculate_metrics(model, dataloader, device):
    """Вычисление Precision, Recall, F1 для мультиклассовой задачи"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Макро-усреднённые метрики (равный вес для каждого класса)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return precision, recall, f1

def plot_training_history(history, save_path):
    """Визуализация истории обучения"""
    plt.figure(figsize=(12, 5))
    
    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()