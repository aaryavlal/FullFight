import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd

# --------------------------------------------
# Config
# --------------------------------------------

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------
# Create Flask App
# --------------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# --------------------------------------------
# Routes
# --------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------
# Parse fight query
# ----------------------------

@app.route('/parse_fight', methods=['POST'])
def parse_fight():
    data = request.json
    query = data.get('query', '')

    # Dummy parser
    if 'vs' in query.lower():
        parts = query.lower().split('vs')
        fighter1 = parts[0].strip().title()
        fighter2 = parts[1].strip().title()
        anime = "Unknown"  # you can improve this later
        return jsonify({'anime': anime, 'fighters': [fighter1, fighter2]})
    else:
        return jsonify({'anime': '', 'fighters': []})

# ----------------------------
# Upload episode videos
# ----------------------------

@app.route('/upload_episodes', methods=['POST'])
def upload_episodes():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            saved_files.append(path)

    return jsonify({'saved_files': saved_files})

# ----------------------------
# Get timestamps
# ----------------------------

@app.route('/get_timestamps', methods=['POST'])
def get_timestamps():
    data = request.json
    anime = data.get('anime')
    fighters = data.get('fighters')

    # Placeholder timestamps
    dummy_timestamps = [
        {"episode": "episode1.mp4", "start": "00:30", "end": "01:45"},
        {"episode": "episode2.mp4", "start": "00:10", "end": "01:15"}
    ]

    return jsonify({'timestamps': dummy_timestamps})

# ----------------------------
# Compile fight video
# ----------------------------

@app.route('/compile_fight', methods=['POST'])
def compile_fight():
    data = request.json
    timestamps = data.get('timestamps')
    video_files = data.get('video_files')

    # Here you'd invoke moviepy or video_editor.py
    # For now, simulate output:
    output_file = "final_fight.mp4"

    return jsonify({'output_file': output_file})

# ----------------------------
# Serve compiled video
# ----------------------------

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# ----------------------------
# CSV endpoints from notebook
# ----------------------------

@app.route('/get_audio_rms')
def get_audio_rms():
    try:
        df = pd.read_csv('audio_rms.csv')
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_brightness')
def get_brightness():
    try:
        df = pd.read_csv('frame_brightness.csv')
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_dialogue')
def get_dialogue():
    try:
        df = pd.read_csv('angry_sections.csv')
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_merged_features')
def get_merged_features():
    try:
        df = pd.read_csv('merged_features.csv')
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)})

# --------------------------------------------
# Run
# --------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
