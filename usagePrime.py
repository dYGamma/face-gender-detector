import numpy as np
import cv2
import os
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import joblib
import streamlit as st

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
        return face, (x, y, w, h)
    return None, None

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
model_save_path = 'learned_model/svm_gender_classifier.pkl'
svm_model = joblib.load(model_save_path)

# Функция для предсказания пола на основе изображения
def predict_gender(image):
    face, box = align_face(image)
    if face is not None:
        embedding = get_embedding(face)
        prediction = svm_model.predict([embedding])
        gender = "Male" if prediction[0] == 1 else "Female"
        return gender, box
    else:
        raise ValueError("Лицо не разпознано")

# Функция для рисования прямоугольника и надписи
def draw_label(image, box, gender):
    x, y, w, h = box
    label = gender
    color = (0, 255, 0) if gender == "Male" else (255, 0, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Создание веб-приложения с помощью Streamlit
st.title("Нейросеть для распознавания пола")
st.write("Загрузите изображение и модель определит ваш пол.")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
    # Преобразование изображения из BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        gender, box = predict_gender(image)
        draw_label(image_rgb, box, gender)
        st.image(image_rgb, caption="Ваше изображение", use_column_width=True)
        gender_rus = "Мужской пол" if gender == "Male" else "Женский пол"
        st.write(f"Гендер человека на основе изображения: {gender_rus}")
    except ValueError as e:
        st.image(image_rgb, caption="Загруженное изображение", use_column_width=True)
        st.write(f"Error: {str(e)}")
