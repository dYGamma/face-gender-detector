from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Определение вашего пользовательского слоя
class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            return [K.cast_to_floatx(x) * self.scale for x in inputs]
        else:
            return inputs * self.scale

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return [shape for shape in input_shape]
        else:
            return input_shape

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

# Загрузка модели с указанием custom_objects
model = load_model('gender_classification_model.h5', custom_objects={'CustomScaleLayer': CustomScaleLayer})

# Компиляция модели
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path, target_size):
    # Загрузка изображения
    img = load_img(image_path, target_size=target_size)
    # Преобразование в numpy массив
    img_array = img_to_array(img)
    # Добавление размерности для соответствия форме входных данных модели
    img_array = np.expand_dims(img_array, axis=0)
    # Нормализация пиксельных значений, если это необходимо
    img_array = img_array / 255.0
    return img_array

# Путь к изображению, которое вы хотите протестировать
image_path = 'nm0000100_rm46373120_1955-1-6_2003.jpg'

# Подготовка изображения
processed_image = preprocess_image(image_path, target_size=(160, 160))  # Измените target_size в соответствии с вашей моделью

# Выполнение предсказания
prediction = model.predict(processed_image)

# Предполагается, что ваша модель имеет один выходной узел с сигмоидной активацией
predicted_class = 'male' if prediction[0] > 0.5 else 'female'

# Загрузка оригинального изображения с использованием OpenCV
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Использование предобученного классификатора Haar для обнаружения лица
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Рисование квадратиков вокруг лица и метки пола
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, predicted_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Преобразование цвета изображения из BGR (используемого OpenCV) в RGB (используемого Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Отображение изображения
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
