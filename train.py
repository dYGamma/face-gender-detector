import numpy as np
import cv2
import os
import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Установка использования GPU для TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Загрузка модели FaceNet из .pb файла
def load_facenet_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    return tf.compat.v1.get_default_graph()


# Путь к .pb файлу
model_path = '20180402-114759/20180402-114759.pb'

# Загрузка модели
graph = load_facenet_model(model_path)

# Получение тензоров ввода и вывода
input_tensor = graph.get_tensor_by_name("input:0")
output_tensor = graph.get_tensor_by_name("embeddings:0")
phase_train_tensor = graph.get_tensor_by_name("phase_train:0")

# Инициализация MTCNN для выравнивания лиц
detector = MTCNN()


# Функция для выравнивания лица
def align_face(image):
    results = detector.detect_faces(image)
    if results:
        x, y, w, h = results[0]['box']
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (160, 160))
        return face
    return None


# Функция для извлечения эмбеддингов с помощью FaceNet
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    with tf.compat.v1.Session(graph=graph) as sess:
        feed_dict = {input_tensor: samples, phase_train_tensor: False}
        embedding = sess.run(output_tensor, feed_dict=feed_dict)
    return embedding[0]


# Функция для обработки одного изображения
def process_image(image_path, gender):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        face = align_face(image)
        if face is not None:
            embedding = get_embedding(face)
            return embedding, int(gender)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    return None, None


# Функция для создания tf.data.Dataset
def create_dataset(dataset_path, metadata_file, batch_size=32, fraction=0.1, max_workers=6):
    data = scipy.io.loadmat(metadata_file)
    full_path = data['imdb'][0][0][2][0]
    gender = data['imdb'][0][0][3][0]

    # Используем только часть данных
    num_samples = int(len(full_path) * fraction)
    full_path = full_path[:num_samples]
    gender = gender[:num_samples]

    paths_and_labels = [(os.path.join(dataset_path, full_path[i][0]), gender[i]) for i in range(len(full_path)) if
                        not np.isnan(gender[i])]

    def generator():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, image_path, gender) for image_path, gender in paths_and_labels]
            for future in tqdm(futures, desc="Processing images"):
                embedding, gender = future.result()
                if embedding is not None:
                    yield embedding, gender

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(512,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    ))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# Путь к датасету и метаданным
dataset_path = 'imdb_crop'  # Путь к папке imdb_crop
metadata_file = os.path.join(dataset_path, 'imdb.mat')  # Путь к файлу imdb.mat

# Создание tf.data.Dataset
print("Creating dataset...")
dataset = create_dataset(dataset_path, metadata_file, batch_size=32, fraction=0.1,
                         max_workers=6)
print("Dataset created.")

# Загрузка данных в numpy массивы
X, y = [], []
for batch in tqdm(dataset, desc="Loading data"):
    X_batch, y_batch = batch
    X.extend(X_batch.numpy())
    y.extend(y_batch.numpy())

X = np.array(X)
y = np.array(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение SVM с прогресс-баром
print("Training SVM model...")
model = SVC(kernel='linear', verbose=True)
model.fit(X_train, y_train)
print("Model trained.")

# Путь для сохранения модели
model_save_path = 'svm_gender_classifier.pkl'

# Сохранение модели
joblib.dump(model, model_save_path)
print(f"Model saved to {model_save_path}")

# Загрузка модели
model = joblib.load(model_save_path)
print("Model loaded.")

# Оценка модели с прогресс-баром
print("Evaluating model...")
y_pred = []
for i in tqdm(range(len(X_test)), desc="Predicting"):
    y_pred.append(model.predict([X_test[i]]))
y_pred = np.array(y_pred).flatten()

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')