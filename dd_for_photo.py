'Тестирование на одном изображении'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
from tensorflow import keras

# Задайте путь к изображению
picture = 'dd_test.jpg'

# Загрузка и настройка модели
base_model = tf.keras.models.load_model('./model/local_inception_model.keras')
dream_model = tf.keras.Model(inputs=base_model.input, outputs=[
    base_model.get_layer(name).output for name in ['mixed3', 'mixed5']
])

# Получите размер изображения и выведите его в консоль
img = Image.open(picture)
width, height = img.size
print(f"Размер изображения перед обработкой: {width}x{height} пикселей")

# Функция для предварительной обработки изображения с использованием полученных размеров
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((width, height))  # Автоматически подставляем размер изображения
    img = np.array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Функция для обратного преобразования изображения из тензора
def deprocess_image(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)

# Функция для расчета потерь DeepDream
def calc_loss(img, model):
    img = tf.expand_dims(img, axis=0)
    layer_activations = model(img)
    losses = [tf.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

# Шаг DeepDream с использованием градиентов
@tf.function
def deepdream_step(img, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    grads = tape.gradient(loss, img)
    grads /= tf.math.reduce_std(grads) + 1e-8
    img = img + grads * step_size
    img = tf.clip_by_value(img, -1, 1)
    return img

# Основная функция DeepDream
def run_deep_dream(image_path, steps=100, step_size=0.01):
    img = preprocess_image(image_path)
    for step in range(steps):
        img = deepdream_step(img, dream_model, step_size)
        if step % 10 == 0:
            display.clear_output(wait=True)
            display.display(Image.fromarray(deprocess_image(img).numpy()))
            print(f"Step {step}, Loss: {calc_loss(img, dream_model).numpy()}")
    return deprocess_image(img)

# Применяю DeepDream к изображению и сохраняю результат
deep_dream_img = run_deep_dream(picture)
result_img = Image.fromarray(deep_dream_img.numpy())
result_img.show()
result_img.save('deepdream_result.jpg')
print("Изображение сохранено как 'deepdream_result.jpg'")