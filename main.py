import cv2
import os
import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import tensorflow as tf
from tensorflow import keras
import time

# Загружаем модель для DeepDream
base_model = tf.keras.models.load_model('./model/local_inception_model.keras')
dream_model = tf.keras.Model(inputs=base_model.input, outputs=[
    base_model.get_layer(name).output for name in ['mixed3', 'mixed5']
])

# Функция для предварительной обработки кадра
def preprocess_image(img):
    img = np.array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Функция для обратного преобразования кадра
def deprocess_image(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)

# Функция DeepDream для кадра
@tf.function
def deepdream_step(img, model, step_size=0.01):
    with tf.GradientTape() as tape:
        tape.watch(img)
        layer_activations = model(img)
        loss = tf.reduce_sum([tf.reduce_mean(act) for act in layer_activations])
    grads = tape.gradient(loss, img)
    grads /= tf.math.reduce_std(grads) + 1e-8
    img = img + grads * step_size
    img = tf.clip_by_value(img, -1, 1)
    return img

def run_deep_dream(img, steps=50, step_size=0.01):
    img = preprocess_image(img)
    img = tf.expand_dims(img, axis=0)  # Добавляем измерение батча
    for _ in range(steps):
        img = deepdream_step(img, dream_model, step_size)
    img = tf.squeeze(img, axis=0)  # Убираем измерение батча после обработки
    return deprocess_image(img).numpy()

# Разделение видео на кадры
cam = cv2.VideoCapture('./video.mp4')
input_fps = cam.get(cv2.CAP_PROP_FPS)
total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Исходная частота кадров: {input_fps}")
print(f"Общее количество кадров в видео: {total_frames}")

# Запрос частоты кадров у пользователя с ограничением до 60
desired_fps = int(input("Введите желаемую частоту кадров для выходного видео (макс 60): "))
if desired_fps >= 60:
    print("Частота кадров ограничена до 60 FPS.")
    desired_fps = 60

print(f"Будет обработано {total_frames} кадров с частотой {desired_fps} кадров/с")

if not os.path.exists('deep'):
    os.makedirs('deep')

currentframe = 0
frames = []
start_time = time.time()  # Начинаем отслеживание общего времени на обработку

while True:
    ret, frame = cam.read()
    if not ret:
        break

    print(f'Обрабатываю кадр {currentframe + 1} из {total_frames}')  # Отображение номера текущего кадра

    # Применяю DeepDream к кадру
    dream_img = run_deep_dream(Image.fromarray(frame))

    # Сохраняю обработанный кадр в список
    frames.append(Image.fromarray(dream_img))

    currentframe += 1

# Освобождаем ресурсы видео
cam.release()
cv2.destroyAllWindows()

# Вычисляем общее время обработки видео
end_time = time.time()
total_duration = end_time - start_time
print(f'Общее время обработки видео: {total_duration:.2f} секунд')

# Собираем обработанные кадры в видео с заданной частотой кадров
clip = ImageSequenceClip([np.array(img) for img in frames], fps=desired_fps)
clip.write_videofile("deep_video.mp4", fps=desired_fps, audio_bitrate="1000k", bitrate="4000k")
print("Видео обработано и сохранено как 'deep_video.mp4'")