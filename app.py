import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import traceback

from moviepy.editor import VideoFileClip, concatenate_videoclips


from full import full_fight_scene_pipeline

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def time_str_to_seconds(time_str):
    parts = [int(p) for p in time_str.strip().split(':')]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")





UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    print("🖨️  Rendering templates/index.html") 
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/parse_fight', methods=['POST'])
def parse_fight():
    data = request.get_json(force=True)
    query = data.get('query', '')

    if 'vs' in query.lower():
        parts = query.lower().split('vs')
        fighter1 = parts[0].strip().title()
        fighter2 = parts[1].strip().title()
        return jsonify(anime='Unknown', fighters=[fighter1, fighter2])
    return jsonify(anime='', fighters=[])

@app.route('/upload_episodes', methods=['POST'])
def upload_episodes():
    files = request.files.getlist('files')
    if not files:
        return jsonify(error='No files part'), 400

    saved_files = []
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            saved_files.append(filename)

    return jsonify(saved_files=saved_files)

@app.route('/get_timestamps', methods=['POST'])
def get_timestamps():
    data = request.get_json(force=True)
    # TODO: Replace with real timestamp-extraction logic
    episodes = os.listdir(app.config['UPLOAD_FOLDER'])
    timestamps = [
        {'episode': ep, 'start': '00:30', 'end': '01:45'}
        for ep in episodes
    ]
    return jsonify(timestamps=timestamps)

@app.route('/compile_fight', methods=['POST'])
def compile_fight():
    try:
        data = request.get_json(force=True)
        print("Compile request received with data:", data)

        filename = data.get('filename')
        if not filename:
            return jsonify(error='No filename received'), 400

        safe_filename = secure_filename(filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

        if not os.path.exists(video_path):
            return jsonify(error=f"File not found: {video_path}"), 404

        # Get predicted timestamps from ML
        raw_results = full_fight_scene_pipeline(video_path)

        # Open video to get duration
        clip = VideoFileClip(video_path)
        duration = clip.duration

        # Buffer each timestamp ±30s (clamped to video bounds)
        buffer = 30
        timestamps = []
        for t in raw_results:
            start = max(0, t - buffer)
            end = min(duration, t + buffer)
            timestamps.append({'start': start, 'end': end})
            print(f"Clipping: {start:.1f}s → {end:.1f}s")

        if not timestamps:
            return jsonify(error="No fight scenes detected"), 200

        # Extract and concatenate subclips
        subclips = [clip.subclip(ts['start'], ts['end']) for ts in timestamps]
        final = concatenate_videoclips(subclips)

        output_filename = f"final_fight_{safe_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return jsonify({
            "message": "Fight compilation complete.",
            "output_file": output_filename
        })

    except Exception as e:
        print(f"[ERROR] {traceback.format_exc()}")
        return jsonify(error=f"Compile error: {str(e)}"), 500

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# @app.route('/get_audio_rms')
# def get_audio_rms():
#     try:
#         df = pd.read_csv('audio_rms.csv')
#         return jsonify(df.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify(error=str(e)), 500

# @app.route('/get_brightness')
# def get_brightness():
#     try:
#         df = pd.read_csv('frame_brightness.csv')
#         return jsonify(df.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify(error=str(e)), 500

# @app.route('/get_dialogue')
# def get_dialogue():
#     try:
#         df = pd.read_csv('angry_sections.csv')
#         return jsonify(df.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify(error=str(e)), 500

# @app.route('/get_merged_features')
# def get_merged_features():
#     try:
#         df = pd.read_csv('merged_features.csv')
#         return jsonify(df.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify(error=str(e)), 500
    
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
