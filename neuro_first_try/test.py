import tensorflow as tf

# Проверка доступности GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Доступные GPU:")
    for gpu in gpus:
        print(f"- {gpu.name}")
else:
    print("GPU не доступен!")
