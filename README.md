# FullFight.AI

**Automated fight scene extraction from anime episodes using multimodal machine learning.**

FullFight.AI is an end-to-end pipeline that ingests raw anime episode files and outputs compiled highlight reels — no manual clipping required. It fuses four independent signal streams (motion, audio, speech emotion, and brightness) into a unified scene classifier, then cuts and concatenates the detected segments via `ffmpeg`.

---

## How It Works

Fight scenes have a consistent multimodal signature: fast motion, loud audio, angry dialogue, and high contrast frames. FullFight.AI extracts each of these independently, merges them into a feature matrix, and trains a `RandomForestClassifier` on hand-labeled data to detect that signature at scale.

```
Episode File (.mp4)
       │
       ├── Audio RMS          → librosa          → rms features
       ├── Optical Flow       → OpenCV Farneback  → motion magnitude
       ├── Frame Brightness   → OpenCV            → brightness features
       └── Speech Emotion     → Whisper + RoBERTa → anger score
                                      │
                              Merge & Normalize
                                      │
                            RandomForestClassifier
                                      │
                              Fight / No-Fight
                                      │
                              ffmpeg clip + concat
                                      │
                           highlight_reel_output.mp4
```

---

## Features

- **Multimodal fusion** — motion, audio, brightness, and NLP emotion signals combined into a single feature vector per time window
- **Self-supervised labeling** — rule-based thresholds bootstrap initial labels; model is trained on top
- **Transformer emotion detection** — uses `cardiffnlp/twitter-roberta-base-emotion` on Whisper ASR transcripts to detect anger in dialogue
- **Web interface** — Flask-based upload UI; drag, drop, get a highlight reel
- **Interactive analysis** — Jupyter notebook for feature visualization, threshold tuning, and model inspection
- **Zero manual editing** — `ffmpeg` handles all clip extraction and compilation

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Flask, Python |
| Video/Audio | ffmpeg-python, librosa, OpenCV |
| Speech | Whisper (OpenAI), Transformers (HuggingFace) |
| ML | scikit-learn (RandomForestClassifier), pandas, numpy |
| Frontend | HTML5, CSS3, JavaScript |
| Analysis | Jupyter, matplotlib, seaborn |

---

## Quickstart

```bash
git clone https://github.com/aaryavlal/FullFight.git
cd FullFight
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`, upload an episode, and download your highlight reel.

To retrain the model on new data:

```bash
jupyter notebook fullflight.ipynb
# Run feature extraction → labeling → training cells in order
```

---

## Pipeline Detail

### 1. Feature Extraction
Each episode is segmented into fixed time windows. For each window:
- **Audio RMS** — root mean square energy via `librosa`
- **Optical flow** — per-frame motion magnitude using Farneback dense flow
- **Brightness** — mean pixel value of grayscale frames
- **Emotion** — Whisper transcribes audio; `cardiffnlp/twitter-roberta-base-emotion` scores anger probability

### 2. Labeling
A window is labeled `fight = 1` if any threshold is exceeded:
- Anger score > 0.5
- Brightness > 150
- Audio RMS > −20 dB
- Optical flow above empirically tuned threshold

### 3. Training
Features are merged, normalized, and fed into a `RandomForestClassifier`. The trained model is serialized to `rf_fight_scene_model.mkl`.

### 4. Inference & Compilation
`full.py` runs the trained model over new episodes, identifies fight windows, and uses `ffmpeg-python` to extract and concatenate the corresponding clips.

---

## Project Structure

```
FullFight/
├── app.py                      # Flask backend
├── full.py                     # Full inference pipeline
├── fullflight.ipynb            # Feature extraction, labeling, training
├── fullflight2.ip              # Utility functions
├── rf_fight_scene_model.mkl    # Trained model
├── templates/index.html        # Upload UI
├── static/
│   ├── style.css
│   └── upload.js
├── uploads/                    # Incoming episode files
├── output/                     # Generated highlight reels
├── audio_rms.csv
├── frame_brightness.csv
├── optical_flow.csv
├── angry_sections.csv
└── normalized_merged_data.csv
```

---

## Authors

- [Aaryav Lal](https://github.com/aaryavlal)
- Dhyan Soni
- Aditya Srivastava

---

## License

MIT
