# DeepDream_for_Video_and_Photo


## Установка

### 1. Установите Python

Скачайте и установите Python с официального сайта:  
🔗 [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Установите FFmpeg

Скачайте и установите FFmpeg для обработки видео:  
🔗 [https://www.ffmpeg.org](https://www.ffmpeg.org)

### 3. Установка пакетов Python

Установите зависимости, указанные в `requirements.txt`. Для этого выполните в терминале:

```bash
pip3 install -r requirements.txt
```
или

```bash
python3 -m pip install -r requirements.txt
```

## Использование

### Шаг 1
1. [Если хочешь обработать видео] 
Поместите видео, которое хотите обработать, в папку со скриптом main.py и переименуйте его в video.mp4.
2. [Если хочешь обработать фото]
Переместите фото, которое хотите обработать, в папку со скриптом main.py и переименуйте его в dd_test.jpg.

### Шаг 2
1. [Если хочешь обработать видео] 
Выполните в терминале:
```bash
python3 main.py
```
2. [Если хочешь обработать фото]
Выполните в терминале:
```bash
python3 dd_photo.py
```
### Шаг 3
Подождите, пока обработка завершится.Обработанное видео будет сохранено в файл deep_video.mp4. Обработанное фото будет сохранено в файл deepdream_result.jpg

### Шаг 4
После использования для освобождения пространства удалите временные папки:
```bash
rm -rf deep data
```

