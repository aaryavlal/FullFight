# FullFight.AI

## 🔎 Overview

**FullFight.AI** is an end-to-end AI pipeline for extracting, analyzing, and compiling fight scenes from anime episodes. It integrates machine learning, audio/video signal processing, NLP, and a full-stack web platform. Users upload episodes via a Flask-based interface and receive curated fight highlight reels—automatically generated.

---

## 🚀 Features

- **Web Interface**: Upload anime episodes and request fight scene extraction  
- **Feature Extraction**:
  - Audio RMS (librosa)  
  - Frame brightness (OpenCV)  
  - Motion (optical flow via OpenCV Farneback)  
  - Dialogue/emotion (anger detection using Whisper + transformer)
- **Automated Labeling**: Combines rule-based thresholds and ML models to label fight segments  
- **Visualization**: Interactive Jupyter notebook for plotting and feature tuning  
- **Video Compilation**: Clips and compiles fight scenes into highlight reels using ffmpeg  
- **ML Integration**: Trains a `RandomForestClassifier` on self-collected and labeled data

---

## 🛠️ Tech Stack

### Backend

- **Flask** – Web server and API endpoints  
- **Python libraries**:
  - `ffmpeg-python` – video/audio processing  
  - `librosa` – audio RMS extraction  
  - `opencv-python` – brightness and optical flow  
  - `whisper` – speech transcription  
  - `transformers`, `torch` – emotion classification  
  - `pandas`, `numpy` – data manipulation  

### Frontend

- **HTML5/CSS3** – Responsive UI (`templates/index.html`, `static/style.css`)  
- **JavaScript** – File uploads, API calls, and dynamic UI updates (`static/upload.js`)  

### Data Science & ML

- **Jupyter Notebook** – Feature extraction, merging, labeling, visualization, and training  
- **scikit-learn** – `RandomForestClassifier` model  
- **pandas**, **matplotlib**, **seaborn** – Data wrangling and visualization  

---

## 📂 Directory Structure
```
FullFight/
│
├── app.py # Flask backend (upload, processing, endpoints)
├── requirements.txt # Project dependencies
├── templates/
│ └── index.html # Web UI
├── static/
│ ├── style.css # Frontend styles
│ └── upload.js # Frontend logic
├── uploads/ # Uploaded video files
├── output/ # Generated highlight clips
├── fullflight.ipynb # Notebook for extraction, analysis, labeling, modeling
├── fullflight2.ip # Functions to be used in full.py
├── full.py # Full pipeline, utlized trained model, and custom data collection functions
├── audio_rms.csv # Extracted audio features
├── frame_brightness.csv # Extracted brightness features
├── optical_flow.csv # Extracted motion features
├── angry_sections.csv # Extracted emotion features
├── normalized_merged_data.csv # Combined feature set
└── rf_fight_scene_model.mkl # Trained model
```

---

## 🔄 Data Pipeline

1. **Upload Video** – Users upload anime episodes via the web UI  
2. **Feature Extraction** – Notebook extracts audio RMS, brightness, motion, emotion  
3. **Merge & Label** – Combined CSV is labeled using a mix of thresholds and manual annotation  
4. **Visualization** – Features are plotted, inspected, and thresholds are refined  
5. **Model Training** – `RandomForestClassifier` is trained on labeled data  
6. **Video Compilation** – Clips for detected fight scenes are extracted with ffmpeg

---

## 💡 Notes

- **Data Collection**: We manually reviewed and labeled each scene based on emotion, brightness, motion, and audio levels  
- **Emotion Detection:** Uses `cardiffnlp/twitter-roberta-base-emotion` on Whisper transcripts  
- **Motion Estimation:** Utilizes OpenCV Farneback for optical flow magnitude  
- **Modeling:** `RandomForestClassifier` trained on features [RMS, brightness, flow, emotion]  
- **Labeling Rules**: A scene is flagged as “fight” if it satisfies at least one of:
  - Anger score > 0.5  
  - Brightness > 150  
  - RMS > –20 dB  
  - Optical flow above an empirically tuned threshold  

---

## 👨‍💻 Authors

- Aaryav Lal  
- Dhyan Soni  
- Aditya Srivastava

---
