# lab_regnet_dogs_kaggle.py
"""
Классификация пород собак из Stanford Dogs Dataset с использованием архитектуры RegNet.

Эксперименты:
1. Дообучение предобученной модели RegNet с оптимизатором Adam
2. Дообучение предобученной модели RegNet с оптимизатором RAdam
3. Обучение модели RegNet с нуля с оптимизатором Adam

Метрики: Precision, Recall, F1-Score, Accuracy
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np
from torchinfo import summary
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
warnings.filterwarnings('ignore')

# ========== 1. ПАРАМЕТРЫ И НАСТРОЙКИ ==========
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Гиперпараметры
BATCH_SIZE = 32
EPOCHS = 10  # Для дообучения
LEARNING_RATE = 1e-4
NUM_CLASSES = 120  # 120 пород собак

# Пути к данным (Kaggle структура)
DATA_DIR = "./stanford_dogs"  # Главная папка
IMAGES_DIR = os.path.join(DATA_DIR, "images", "Images")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations", "Annotation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")
print(f"Путь к изображениям: {IMAGES_DIR}")

# ========== 2. КАСТОМНЫЙ ДАТАСЕТ ДЛЯ STANFORD DOGS (Kaggle) ==========
class StanfordDogsDataset(Dataset):
    """Кастомный датасет для Stanford Dogs (Kaggle версия)."""
    
    def __init__(self, root_dir, transform=None, use_annotations=False):
        """
        Args:
            root_dir (string): Путь к папке 'images/Images'
            transform (callable, optional): Трансформации для изображений
            use_annotations (bool): Использовать аннотации для кропа
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_annotations = use_annotations
        
        # Собираем список всех изображений и их меток
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Получаем список пород (подпапок)
        breeds = sorted([d for d in os.listdir(root_dir) 
                        if os.path.isdir(os.path.join(root_dir, d))])
        
        # Создаем mapping: имя папки -> индекс класса
        for idx, breed in enumerate(breeds):
            self.class_to_idx[breed] = idx
            breed_dir = os.path.join(root_dir, breed)
            
            # Собираем все изображения в папке породы
            for img_name in os.listdir(breed_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(breed_dir, img_name)
                    self.samples.append((img_path, idx))
        
        self.classes = breeds
        print(f"Найдено {len(self.classes)} пород, {len(self.samples)} изображений")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        
        # Если используем аннотации, кропаем по bounding box
        if self.use_annotations:
            try:
                # Получаем путь к аннотации
                breed_folder = self.classes[label]
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                ann_path = os.path.join(ANNOTATIONS_DIR, breed_folder, img_id)
                
                # Парсим XML аннотацию
                tree = ET.parse(ann_path)
                root = tree.getroot()
                
                # Получаем bounding box
                obj = root.find('object')
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Кропаем изображение
                image = image.crop((xmin, ymin, xmax, ymax))
            except Exception as e:
                print(f"Ошибка при обработке аннотации {img_path}: {e}")
                # Если аннотация не найдена, используем все изображение
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ========== 3. ПОДГОТОВКА ДАННЫХ ==========
# Трансформации
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    normalize
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# Создаем полный датасет (без использования аннотаций для кропа - быстрее)
# Если хотите использовать аннотации, установите use_annotations=True
print("Загрузка датасета...")
full_dataset = StanfordDogsDataset(
    root_dir=IMAGES_DIR,
    transform=None,  # Трансформации применим позже
    use_annotations=False  # Установите True для кропа по аннотациям
)

# Разделение датасета на train/val/test (70/15/15)
dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size  # Остаток идет в тест

# Разделяем индексы с использованием random_split для более точного разделения
indices = list(range(dataset_size))
np.random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

print(f"Разделение данных: Train={len(train_indices)} ({100*len(train_indices)/dataset_size:.1f}%), "
      f"Val={len(val_indices)} ({100*len(val_indices)/dataset_size:.1f}%), "
      f"Test={len(test_indices)} ({100*len(test_indices)/dataset_size:.1f}%)")

# Функция для создания Subset с трансформациями
def create_subset_with_transform(dataset, indices, transform):
    """Создает поднабор датасета с указанными трансформациями."""
    from torch.utils.data import Subset
    
    class TransformedSubset(Subset):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            if transform:
                img = transform(img)
            return img, label
    
    return TransformedSubset(dataset, indices)

# Создаем поднаборы с трансформациями
train_dataset = create_subset_with_transform(full_dataset, train_indices, train_transform)
val_dataset = create_subset_with_transform(full_dataset, val_indices, val_test_transform)
test_dataset = create_subset_with_transform(full_dataset, test_indices, val_test_transform)

# DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                         shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=2, pin_memory=True)

print(f"Размеры выборок: Обучающая: {len(train_dataset)}, "
      f"Валидационная: {len(val_dataset)}, Тестовая: {len(test_dataset)}")

# ========== 4. ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ И ОЦЕНКИ ==========
def train_epoch(model, loader, criterion, optimizer, device):
    """Обучение модели на одной эпохе."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Прогресс
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(all_preds), np.array(all_labels)

def evaluate(model, loader, criterion, device):
    """Оценка модели."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    
    # Также считаем accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return epoch_loss, accuracy, precision, recall, f1, np.array(all_preds), np.array(all_labels)

# ========== 5. ЭКСПЕРИМЕНТ 1: ДООБУЧЕНИЕ PRE-TRAINED RegNet ==========
print("\n" + "="*60)
print("ЭКСПЕРИМЕНТ 1: ДООБУЧЕНИЕ PRE-TRAINED RegNet")
print("="*60)

def create_pretrained_model():
    """Создает предобученную модель RegNet."""
    model = models.regnet_y_3_2gf(weights='DEFAULT')
    
    # Заморозка всех слоев
    for param in model.parameters():
        param.requires_grad = False
    
    # Заменяем последний слой
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    return model

# --- 1A: Дообучение с Adam ---
print("\n--- 1A: Дообучение с Adam ---")
model_adam = create_pretrained_model().to(DEVICE)
print(summary(model_adam, input_size=(BATCH_SIZE, 3, 224, 224)))

criterion = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(model_adam.fc.parameters(), lr=LEARNING_RATE)

history_adam = {'train_loss': [], 'val_loss': [], 'val_acc': [], 
                'val_precision': [], 'val_recall': [], 'val_f1': []}

print("Начинаем дообучение...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Обучение
    train_loss, _, _ = train_epoch(model_adam, train_loader, criterion, optimizer_adam, DEVICE)
    
    # Валидация
    val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
        model_adam, val_loader, criterion, DEVICE)
    
    # Сохраняем историю
    history_adam['train_loss'].append(train_loss)
    history_adam['val_loss'].append(val_loss)
    history_adam['val_acc'].append(val_acc)
    history_adam['val_precision'].append(val_precision)
    history_adam['val_recall'].append(val_recall)
    history_adam['val_f1'].append(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

# Тестирование
print("\nТестирование дообученной модели (Adam)...")
test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = evaluate(
    model_adam, test_loader, criterion, DEVICE)
print(f"Test Results: Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
      f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
results_adam = (test_acc, test_precision, test_recall, test_f1)

# --- 1B: Дообучение с RAdam ---
print("\n--- 1B: Дообучение с RAdam ---")

# Пробуем импортировать RAdam, если не установлен - используем AdamW
try:
    from torch_optimizer import RAdam
    print("RAdam найден, используем его...")
    radam_available = True
except ImportError:
    print("RAdam не найден. Установите: pip install torch_optimizer")
    print("Используем AdamW вместо RAdam...")
    radam_available = False

model_radam = create_pretrained_model().to(DEVICE)

if radam_available:
    optimizer_radam = RAdam(model_radam.fc.parameters(), lr=LEARNING_RATE)
else:
    optimizer_radam = optim.AdamW(model_radam.fc.parameters(), lr=LEARNING_RATE)

history_radam = {'train_loss': [], 'val_loss': [], 'val_acc': [], 
                 'val_precision': [], 'val_recall': [], 'val_f1': []}

print("Начинаем дообучение...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    train_loss, _, _ = train_epoch(model_radam, train_loader, criterion, optimizer_radam, DEVICE)
    val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
        model_radam, val_loader, criterion, DEVICE)
    
    history_radam['train_loss'].append(train_loss)
    history_radam['val_loss'].append(val_loss)
    history_radam['val_acc'].append(val_acc)
    history_radam['val_precision'].append(val_precision)
    history_radam['val_recall'].append(val_recall)
    history_radam['val_f1'].append(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

print("\nТестирование дообученной модели (RAdam)...")
test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = evaluate(
    model_radam, test_loader, criterion, DEVICE)
print(f"Test Results: Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
      f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
results_radam = (test_acc, test_precision, test_recall, test_f1)

# ========== 6. ЭКСПЕРИМЕНТ 2: ОБУЧЕНИЕ С НУЛЯ НА Adam ==========
print("\n" + "="*60)
print("ЭКСПЕРИМЕНТ 2: ОБУЧЕНИЕ RegNet С НУЛЯ")
print("="*60)

# Загружаем модель без предобученных весов
model_scratch = models.regnet_y_3_2gf(weights=None)
num_features = model_scratch.fc.in_features
model_scratch.fc = nn.Linear(num_features, NUM_CLASSES)
model_scratch = model_scratch.to(DEVICE)

# Все параметры обучаются
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE*5)  # Больше LR

history_scratch = {'train_loss': [], 'val_loss': [], 'val_acc': [], 
                   'val_precision': [], 'val_recall': [], 'val_f1': []}
EPOCHS_SCRATCH = 15  # Больше эпох для обучения с нуля

print("Начинаем обучение с нуля...")
for epoch in range(EPOCHS_SCRATCH):
    print(f"\nEpoch {epoch+1}/{EPOCHS_SCRATCH}")
    
    train_loss, _, _ = train_epoch(model_scratch, train_loader, criterion, optimizer_scratch, DEVICE)
    val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
        model_scratch, val_loader, criterion, DEVICE)
    
    history_scratch['train_loss'].append(train_loss)
    history_scratch['val_loss'].append(val_loss)
    history_scratch['val_acc'].append(val_acc)
    history_scratch['val_precision'].append(val_precision)
    history_scratch['val_recall'].append(val_recall)
    history_scratch['val_f1'].append(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

print("\nТестирование модели, обученной с нуля...")
test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = evaluate(
    model_scratch, test_loader, criterion, DEVICE)
print(f"Test Results: Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
      f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
results_scratch = (test_acc, test_precision, test_recall, test_f1)

# ========== 7. ВИЗУАЛИЗАЦИЯ И СРАВНЕНИЕ ==========
print("\n" + "="*60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*60)
print(f"{'Модель':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-"*70)
print(f"{'Pretrained + Adam':<30} {results_adam[0]:<10.4f} {results_adam[1]:<10.4f} "
      f"{results_adam[2]:<10.4f} {results_adam[3]:<10.4f}")
print(f"{'Pretrained + RAdam':<30} {results_radam[0]:<10.4f} {results_radam[1]:<10.4f} "
      f"{results_radam[2]:<10.4f} {results_radam[3]:<10.4f}")
print(f"{'Scratch + Adam':<30} {results_scratch[0]:<10.4f} {results_scratch[1]:<10.4f} "
      f"{results_scratch[2]:<10.4f} {results_scratch[3]:<10.4f}")

# Графики
plt.figure(figsize=(15, 10))

# 1. Графики потерь
plt.subplot(2, 3, 1)
plt.plot(history_adam['train_loss'], label='Pretrained Adam Train', linewidth=2)
plt.plot(history_adam['val_loss'], label='Pretrained Adam Val', linewidth=2)
plt.title('Pretrained Adam: Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(history_radam['train_loss'], label='Pretrained RAdam Train', linewidth=2)
plt.plot(history_radam['val_loss'], label='Pretrained RAdam Val', linewidth=2)
plt.title('Pretrained RAdam: Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(history_scratch['train_loss'], label='Scratch Train', linewidth=2)
plt.plot(history_scratch['val_loss'], label='Scratch Val', linewidth=2)
plt.title('Scratch: Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Графики Accuracy
plt.subplot(2, 3, 4)
plt.plot(history_adam['val_acc'], label='Pretrained Adam', linewidth=2)
plt.plot(history_radam['val_acc'], label='Pretrained RAdam', linewidth=2)
plt.plot(history_scratch['val_acc'], label='Scratch', linewidth=2)
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 3. Графики Precision/Recall
plt.subplot(2, 3, 5)
plt.plot(history_adam['val_precision'], label='Adam Precision', linewidth=2)
plt.plot(history_adam['val_recall'], '--', label='Adam Recall', linewidth=2)
plt.plot(history_radam['val_precision'], label='RAdam Precision', linewidth=2)
plt.plot(history_radam['val_recall'], '--', label='RAdam Recall', linewidth=2)
plt.title('Validation Precision & Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# 4. Сравнение F1-Score
plt.subplot(2, 3, 6)
models_names = ['Pretrained\nAdam', 'Pretrained\nRAdam', 'Scratch\nAdam']
f1_scores = [results_adam[3], results_radam[3], results_scratch[3]]
bars = plt.bar(models_names, f1_scores, color=['blue', 'orange', 'green'])
plt.title('F1-Score on Test Set')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
# Добавляем значения на столбцы
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('training_results_kaggle.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nГрафики сохранены в 'training_results_kaggle.png'")

# Сохранение моделей (опционально)
torch.save(model_adam.state_dict(), 'pretrained_adam.pth')
torch.save(model_radam.state_dict(), 'pretrained_radam.pth')
torch.save(model_scratch.state_dict(), 'scratch_adam.pth')
print("\nМодели сохранены в текущей директории.")