# DeepDream_for_Video_and_Photo

## Installation

### 1. Install Python

Download and install Python from the official website:  
🔗 [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Install FFmpeg

Download and install FFmpeg for video processing:  
🔗 [https://www.ffmpeg.org](https://www.ffmpeg.org)

### 3. Install Python Packages

Install the dependencies listed in `requirements.txt`. To do this, run in the terminal:

```bash
pip3 install -r requirements.txt
```

## Usage

### Step 1
1.	[To process a video]
Place the video you want to process in the folder with main.py and rename it to video.mp4.
2. [To process a photo]
Place the photo you want to process in the folder with main.py and rename it to dd_test.jpg.

### Step 2
1. [To process a video]
Run the following command in the terminal:
```bash
python3 main.py
```
2. [To process a photo]
Run the following command in the terminal:
```bash
python3 dd_photo.py
```

### Step 3
Wait until the processing is complete. The processed video will be saved as deep_video.mp4. The processed photo will be saved as deepdream_result.jpg.

### Step 4
After using the project, delete temporary folders to free up space:
```bash
rm -rf deep data
```
