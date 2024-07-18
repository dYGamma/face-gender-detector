import numpy as np
import cv2
import os
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import joblib

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


# Загрузка модели SVM
model_save_path = '77-80%/svm_gender_classifier.pkl'
svm_model = joblib.load(model_save_path)


# Функция для предсказания пола на основе изображения
def predict_gender(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    face = align_face(image)
    if face is not None:
        embedding = get_embedding(face)
        prediction = svm_model.predict([embedding])
        return "Male" if prediction[0] == 1 else "Female"
    else:
        raise ValueError("No face detected in the image")


# Пример использования
image_path = 'IMG_20240630_192348.jpg'
gender = predict_gender(image_path)
print(f"The predicted gender is: {gender}")