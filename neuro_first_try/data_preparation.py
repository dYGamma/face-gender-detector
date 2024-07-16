# функции для загрузки и предобработки данных
import os
import scipy.io
import cv2
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_data_batch(mat_file, img_dir, batch_size=1000):
    logging.info("Загрузка данных из .mat файла: %s", mat_file)
    mat = scipy.io.loadmat(mat_file)
    data = mat['imdb'][0,0]
    image_paths = data['full_path'][0]
    genders = data['gender'][0]
    
    num_samples = len(image_paths)
    X = []
    y = []
    
    for i in range(0, num_samples, batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        batch_genders = genders[i:i + batch_size]
        
        batch_X = []
        batch_y = []
        
        for img_path, gender in zip(batch_image_paths, batch_genders):
            if np.isnan(gender):  # Skip if gender is NaN
                continue
            full_img_path = os.path.join(img_dir, img_path[0])
            img = cv2.imread(full_img_path)
            if img is not None:
                img = cv2.resize(img, (160, 160))  # Resize to 160x160
                batch_X.append(img)
                batch_y.append(int(gender))
            else:
                logging.warning("Не удалось загрузить изображение: %s", full_img_path)
        
        if batch_X:
            X.extend(batch_X)
            y.extend(batch_y)
        
        logging.info("Завершена загрузка и обработка батча изображений. Батч %d/%d", i // batch_size + 1, (num_samples // batch_size) + 1)
    
    logging.info("Завершена загрузка и обработка всех изображений.")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    imdb_mat_file = 'E:/neuro2/imdb_crop/imdb.mat'
    imdb_img_dir = 'E:/neuro2/imdb_crop'

    logging.info("Начало загрузки и предобработки данных...")
    X, y = load_data_batch(imdb_mat_file, imdb_img_dir)

    logging.info("Сохранение данных в .npy файлы...")
    np.save('X.npy', X)
    np.save('y.npy', y)

    logging.info("Процесс подготовки данных завершен.")