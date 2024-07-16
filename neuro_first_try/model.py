import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import InceptionResNetV2

def create_model(input_shape=(160, 160, 3)):
    base_model = InceptionResNetV2(input_shape=input_shape, include_top=False)
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Создание и компиляция модели
model = create_model()
model.summary()
