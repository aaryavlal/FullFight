import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import traceback

from moviepy.editor import VideoFileClip, concatenate_videoclips



def time_str_to_seconds(time_str):
    """Convert a time string 'MM:SS' or 'HH:MM:SS' to seconds."""
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
        
        video_files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not video_files:
            return jsonify(error='No video file uploaded'), 400

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
        clip = VideoFileClip(video_path)

  
        timestamps = [
            {"start": "00:00", "end": "04:57"},
            {"start": "05:18", "end": "10:11"},
            {"start": "12:09", "end": "15:14"},
            {"start": "20:51", "end": "23:24"}
        ]


        subclips = []
        for ts in timestamps:
            start_sec = time_str_to_seconds(ts["start"])
            end_sec = time_str_to_seconds(ts["end"])
            subclip = clip.subclip(start_sec, end_sec)
            subclips.append(subclip)

        # Stitch together
        final = concatenate_videoclips(subclips)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'final_fight.mp4')
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return jsonify(output_file='final_fight.mp4')

    except Exception as e:
        print(f"[❌ ERROR] {e}")
        return jsonify(error=f"Compile error: {e}"), 500

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
