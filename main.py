
import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_optimizer import RAdam  # Специальная библиотека для RAdam
from utils import get_data_loaders, calculate_metrics, plot_training_history

# Определение устройства (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Параметры обучения
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DATA_DIR = "data/stanford_dogs"  # Путь к распакованному датасету
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"

# Создание директорий
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_optimizer import RAdam
from utils import get_data_loaders, calculate_metrics, plot_training_history

# Отключаем предупреждение о символических ссылках
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Определение устройства (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Если используется CPU, уменьшаем размер батча для ускорения
BATCH_SIZE = 32 if device.type == 'cuda' else 16
print(f"Размер батча: {BATCH_SIZE}")

# Параметры обучения
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DATA_DIR = "data/stanford_dogs"  # Путь к распакованному датасету
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"

# Создание директорий
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_model(pretrained=True, num_classes=120):
    """Создание модели RegNet с возможностью предобучения"""
    # Используем RegNetX-200MF - компактная версия для быстрого обучения
    model = timm.create_model('regnetx_002', pretrained=pretrained)
    
    # Замена последнего слоя под нужное количество классов
    # В новых версиях timm последний слой находится в model.head.fc
    print("Структура модели RegNet:")
    print(f"- Наличие атрибута 'head': {hasattr(model, 'head')}")
    if hasattr(model, 'head'):
        print(f"- Тип head: {type(model.head)}")
        if hasattr(model.head, 'fc'):
            print("- Используется model.head.fc для замены последнего слоя")
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model.head, 'in_features'):
            print("- Используется model.head для замены последнего слоя")
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        else:
            # Для некоторых версий timm head может быть Sequential
            print("- Пытаемся найти последний Linear слой в model.head")
            for name, module in model.head.named_modules():
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    model.head = nn.Linear(in_features, num_classes)
                    break
    else:
        # Резервные варианты для других архитектур
        if hasattr(model, 'fc'):
            print("- Используется model.fc для замены последнего слоя (устаревший вариант)")
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            print("- Используется model.classifier для замены последнего слоя")
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Не удалось определить последний слой для замены. Проверьте структуру модели.")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """Функция обучения модели с сохранением истории"""
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nЭпоха {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Словарь для хранения результатов каждой фазы
        epoch_results = {'train': {}, 'val': {}}
        
        # Режимы обучения и валидации
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Итерация по данным
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Обнуление градиентов
                optimizer.zero_grad()
                
                # Прямой и обратный проход только в режиме обучения
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            epoch_results[phase] = {'loss': epoch_loss, 'acc': epoch_acc.item()}
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Сохранение лучшей модели
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
        
        # Обновление learning rate после завершения эпохи (после валидации)
        # ReduceLROnPlateau требует метрику для оценки
        if scheduler is not None:
            # Передаем валидационную точность (чем выше, тем лучше)
            scheduler.step(epoch_results['val']['acc'])
    
    print(f'Лучшая точность валидации: {best_val_acc:.4f}')
    return history
    
def experiment(pretrained=True, optimizer_name='adam', epochs=15):
    """Проведение эксперимента с заданными параметрами"""
    # Загрузка данных
    print("\nЗагрузка данных...")
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE,
        val_size=0.15,
        test_size=0.15
    )
    
    # Создание модели
    print("\nСоздание модели...")
    num_classes = len(class_to_idx)
    model = create_model(pretrained=pretrained, num_classes=num_classes)
    
    # Оптимизатор
    if optimizer_name.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=LEARNING_RATE)
        opt_name = "RAdam"
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        opt_name = "Adam"
    
    # Планировщик learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )
    
    # Функция потерь
    criterion = nn.CrossEntropyLoss()
    
    # Обучение
    print(f"\nНачало обучения (предобучение={pretrained}, оптимизатор={opt_name})")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=epochs
    )
    
    # Тестирование
    print("\nТестирование модели...")
    precision, recall, f1 = calculate_metrics(model, test_loader, device)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Визуализация
    plot_name = f"history_pretrained_{pretrained}_opt_{opt_name}.png"
    plot_training_history(history, os.path.join(RESULTS_DIR, plot_name))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'history': history,
        'model': model
    }

if __name__ == "__main__":
    # Эксперимент 1: Предобученная модель + Adam
    print("="*50)
    print("ЭКСПЕРИМЕНТ 1: Предобученная RegNet + Adam")
    print("="*50)
    exp1 = experiment(pretrained=True, optimizer_name='adam', epochs=NUM_EPOCHS)
    
    # Эксперимент 2: Предобученная модель + RAdam
    print("="*50)
    print("ЭКСПЕРИМЕНТ 2: Предобученная RegNet + RAdam")
    print("="*50)
    exp2 = experiment(pretrained=True, optimizer_name='radam', epochs=NUM_EPOCHS)
    
    # Эксперимент 3: Модель с нуля + Adam
    print("="*50)
    print("ЭКСПЕРИМЕНТ 3: RegNet с нуля + Adam")
    print("="*50)
    exp3 = experiment(pretrained=False, optimizer_name='adam', epochs=NUM_EPOCHS)
    
    # Сравнение результатов
    results = {
        "Pretrained + Adam": (exp1['precision'], exp1['recall']),
        "Pretrained + RAdam": (exp2['precision'], exp2['recall']),
        "From Scratch + Adam": (exp3['precision'], exp3['recall'])
    }
    
    # Визуализация сравнения
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    precisions = [v[0] for v in results.values()]
    recalls = [v[1] for v in results.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, precisions, width, label='Precision')
    rects2 = ax.bar(x + width/2, recalls, width, label='Recall')
    
    ax.set_ylabel('Значение метрик')
    ax.set_title('Сравнение моделей по Precision и Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    
    # Добавление значений над столбцами
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'))
    plt.close()
    
    print("\nВсе эксперименты завершены!")
    print("Результаты сохранены в папке 'results/'")