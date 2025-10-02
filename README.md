# 🌳 TreeScan AI - Intelligent Tree Health Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-powered system for automated tree detection and health analysis**

*Detect multiple trees with YOLOv9 • Analyze each tree individually • Get comprehensive health reports*

</div>

## 📋 Оглавление

- [🌟 Возможности](#-возможности)
- [🛠 Технологический стек](#-технологический-стек)
- [🚀 Быстрый старт](#-быстрый-старт)
- [📁 Структура проекта](#-структура-проекта)
- [🔧 Установка и настройка](#-установка-и-настройка)
- [💻 Использование](#-использование)

## 🌟 Возможности

### 📈 Комплексный анализ здоровья
- **Оценка здоровья**: балльная система от 1 до 10
- **Идентификация вида**: определение породы дерева
- **Анализ дефектов**: трещины, гниль, дупла, повреждения коры
- **Процент сухих ветвей**: категориальная оценка
- **Угол наклона**: измерение наклона ствола

### 🖼️ Мульти-форматная поддержка
- **Одиночные изображения** с обработкой VLM
- **Альбомы фотографий** (обработка группой)
- **Видео анализ** (последовательный анализ кадров) c VideoChat 2.5



## 🛠 Технологический стек

### 🤖 Модели ИИ
- **Vision-Language Model (VLM)**: Анализ изображений и генерация текста
- **Transformers**: Работа с языковыми моделями

### 🖥 Backend
- **Python 3.8+**: Основной язык программирования
- **PyTorch**: Фреймворк глубокого обучения
- **PIL/Pillow**: Обработка изображений
- **OpenCV**: Компьютерное зрение

### 📱 Интерфейс
- **python-telegram-bot**: Telegram Bot API
- **asyncio**: Асинхронная обработка
- **io.BytesIO**: Работа с бинарными данными

### 🎯 Дополнительные библиотеки
- **decord**: Декодирование видео
- **exifread**: Чтение метаданных
- **numpy**: Научные вычисления

## 🚀 Быстрый старт

### Предварительные требования
- Python 3.8 или выше
- GPU с поддержкой CUDA (рекомендуется)
- Активный Telegram бот

### Установка за 5 минут

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/TreeScan-AI.git
cd TreeScan-AI

# Создание виртуального окружения
conda create -n cv python=3.10
conda activate cv
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
echo "TELEGRAM_BOT_TOKEN=your_bot_token_here" > .env

# Запуск бота
python bot.py
```
## 📁 Структура проекта
```bash
TreeScan-AI/
├── 🤖 bot.py                 # Основной файл бота Telegram
├── 🔍 detection_utils.py     # Детекция деревьев (YOLOv9)
├── 📊 analysis_utils.py      # Анализ здоровья деревьев (VLM)
├── 🛠 model_utils.py         # Загрузка и управление моделями
├── 🖼 image_utils.py         # Утилиты обработки изображений
├── 🎥 video_utils.py         # Обработка видео файлов
├── ⚙️ config.py              # Конфигурация и константы
├── 📋 requirements.txt       # Зависимости проекта
└── 📄 README.md             # Документация
'''
## 🔧 Установка и настройка
1. Установка зависимостей
```bash
# Основные зависимости
pip install python-telegram-bot pillow
pip install transformers opencv-python
pip install decord exifread numpy transformers

# Или установка из requirements.txt
pip install -r requirements.txt
```

2. Настройка Telegram бота
```bash
# Создайте бота через @BotFather в Telegram
# Получите токен и добавьте в .env файл
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHI***********

# Или напрямую в коде (не рекомендуется для продакшена)
TOKEN = "your_actual_token_here"
```

3. Конфигурация моделей
```bash
# В config.py настройте пути к моделям
MODEL_PATH = "path/to/your/vlm/model"
YOLO_MODEL_PATH = "path/to/yolov9.pt"

# Настройки обработки изображений
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB
MAX_PIXELS = 25_000_000           # 25MP
```

## 💻 Использование  
Команды бота  
```bash
/start - Начало работы с ботом
```

Процесс анализа
Отправьте изображение в Telegram бот

Индивидуальный анализ - изображение анализируется VLM

Генерация отчетов - создание таблиц и детальных описаний

Отправка результатов - пользователь получает полный анализ

Пример взаимодействия
```bash
Пользователь: [отправляет фото леса]

Бот: 🌳 Выполняю детальный анализ...
```


Детальный анализ

Вид и порода: Клен. Тип: дерево  
Здоровье: 9 баллов из 10    
Сухие ветви: 0-25%  
Угол наклона: 5 градусов  
Дефекты: Дефектов не обнаружено  

Дополнительные проверки:
✅ Механические повреждения: Нет  
⚠️ Трещины: Да, мелкие поверхностные  
✅ Гниль: Нет  
✅ Отслоение коры: Нет  
✅ Обширное дупло: Нет  
✅ Обширная сухобочина: Нет  
🎯 Детали реализации  
Архитектура системы

```bash
# Основной поток обработки
1. Получение изображения → 2. Детекция YOLOv9 → 
3. Кропинг bounding boxes → 4. VLM анализ каждого кропа → 
5. Агрегация результатов → 6. Генерация отчетов → 
7. Визуализация → 8. Отправка пользователю
Модуль детекции (YOLOv9)
```

Обработка различных форматов

```bash
# Одиночные изображения - полный анализ с детекцией
# Альбомы - пакетная обработка без детекции  
# Видео - анализ ключевых кадров
```



👥 Авторы
Fatik_AI

<div align="center">
TreeScan AI - делаем мониторинг здоровья деревьев доступным для всех! 🌳✨
