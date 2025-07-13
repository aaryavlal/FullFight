# 🥊 FullFight.AI

## 🔎 Overview

**FullFight.AI** is an end-to-end pipeline for extracting, analyzing, and compiling fight scenes from anime episodes using machine learning, audio/video processing, and NLP. The project includes a Flask web backend, a modern HTML/CSS/JS frontend, and a Jupyter notebook for feature extraction and labeling.

---

## 🚀 Features

- **🌐 Web Interface:** Upload anime episodes and submit fight queries.
- **🎞️ Audio/Video Feature Extraction:** Extracts frame brightness, audio RMS, and dialogue/emotion features.
- **🗣️ Dialogue & Emotion Analysis:** Uses Whisper and a transformer-based emotion classifier to detect angry dialogue.
- **🔗 Data Merging & Labeling:** Merges features into a single CSV and labels fight scenes.
- **📊 Visualization:** Jupyter notebook cells for plotting and exploring features.
- **✂️ Video Editing:** Clips and compiles fight scenes based on detected timestamps.
- **🤖 ML Model Integration:** Prepares labeled data for downstream modeling.

---

## 🛠️ Tech Stack

### 🐍 Backend

- **Flask:** Web server and API endpoints.
- **Python Libraries:**
  - `ffmpeg-python` — Video/audio processing.
  - `librosa` — Audio analysis (RMS).
  - `opencv-python` — Frame extraction and brightness calculation.
  - `transformers`, `torch` — NLP and emotion classification.
  - `whisper` — Speech-to-text transcription.
  - `pandas`, `numpy` — Data manipulation and analysis.

### 💻 Frontend

- **HTML5/CSS3:** Responsive UI (`templates/index.html`, `static/style.css`).
- **JavaScript:** Handles file uploads, API calls, and dynamic UI updates.

### 📊 Data Science

- **Jupyter Notebook:** For feature extraction, merging, labeling, and visualization.
- **matplotlib/seaborn:** Data visualization.
- **pandas:** Data wrangling.

---

## 🗂️ Directory Structure

```
FullFight/
│
├── app.py                  # Flask backend
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend HTML
├── static/
│   ├── style.css           # Frontend CSS
│   └── upload.js           # Frontend JS
├── uploads/                # Uploaded video files
├── output/                 # Output video clips
├── utils/
│   └── video_editor.py     # Video clipping utilities
├── ML/
│   └── fullflight.ipynb    # Jupyter notebook for feature extraction & labeling
├── audio_rms.csv           # Audio RMS features
├── frame_brightness.csv    # Frame brightness features
├── angry_sections.csv      # Dialogue/emotion features
├── merged_features.csv     # Merged features
└── labeled_features.csv    # Labeled dataset
```




---

## 📁 Key Files & Their Roles

- **`app.py`** → Main Flask app, handles routes for parsing, uploading, and compiling.
- **`video_editor.py`** → Extracts video clips around fight timestamps.
- **`fullflight.ipynb`** → Notebook for extracting features, merging, labeling, and visualization.
- **`index.html / style.css / upload.js`** → Frontend for user interaction.

---

## 🔄 Data Pipeline

1. **📤 Upload Video:** User uploads anime episode via web UI.
2. **📈 Feature Extraction:** Notebook extracts audio RMS, frame brightness, and dialogue/emotion features.
3. **📝 Merge & Label:** Features are merged and labeled as fight/non-fight.
4. **🔍 Visualization:** Data is visualized for inspection and model training.
5. **✂️ Video Editing:** Clips are extracted around detected fight timestamps.
6. **🧠 Modeling:** Labeled CSV is ready for ML model training.

---

## 💡 Notes

- **Emotion Model:** Uses `cardiffnlp/twitter-roberta-base-emotion` for anger detection.
- **Speech-to-Text:** Uses OpenAI Whisper.
- **Video Editing:** Uses ffmpeg for fast, lossless clipping.
- **Labeling Logic:** Fight scenes are labeled if any of:
  - Segment Anger Score > 0.5
  - Brightness > 150
  - RMS > -20

---

## 👨‍💻 Authors

- Aditya Srivatava
- Aaryav Lal
- Dhyan Soni

---
