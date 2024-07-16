import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_preparation import load_data_batch
from model import create_model
from utils import plot_metrics, evaluate_model
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Проверка доступности GPU и настройка динамического распределения памяти
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logging.info("GPU доступен!")
    try:
        for gpu in gpus:
            # Устанавливаем динамическое распределение памяти для каждого GPU
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Вызов set_memory_growth должен быть сделан до запуска любых операций TensorFlow
        logging.error(e)
else:
    logging.info("GPU не доступен!")

# Путь к датасету
imdb_mat_file = 'E:/neuro2/imdb_crop/imdb.mat'
imdb_img_dir = 'E:/neuro2/imdb_crop'

logging.info("Загрузка и предобработка данных...")
# Загрузка и предобработка данных
X, y = load_data_batch(imdb_mat_file, imdb_img_dir)

logging.info("Разделение данных на обучающую и тестовую выборки...")
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Использование стратегии распределенного обучения
strategy = tf.distribute.MirroredStrategy()
logging.info('Количество устройств: %d', strategy.num_replicas_in_sync)

with strategy.scope():
    logging.info("Создание и компиляция модели...")
    # Создание и компиляция модели
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    logging.info("Начало обучения модели...")
    # Обучение модели
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

logging.info("Сохранение модели...")
# Сохранение модели
model.save('gender_classification_model.h5')

logging.info("Отображение метрик...")
# Отображение метрик
plot_metrics(history)

logging.info("Оценка модели...")
# Оценка модели
evaluate_model(model, X_test, y_test)
