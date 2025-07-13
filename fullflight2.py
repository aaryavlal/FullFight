import librosa
import os
import matplotlib.pyplot as plt
import ffmpeg
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datetime
import whisper
import cv2
import pandas as pd
import glob
from tqdm import tqdm
import csv
import numpy as np

def get_video(name):
    return "uploads/'" + name + "'"

def plot_time(x, y, xlabel, ylabel, title):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

def extract_frames(name, frame_dir="frames"):
    os.makedirs(frame_dir, exist_ok=True)

    output_pattern = os.path.join(
        frame_dir, f"{os.path.basename(name)}_frame_%04d.png"
    )

    print(f"[DEBUG] Extracting frames from: {name}")
    print(f"[DEBUG] Output pattern: {output_pattern}")

    try:
        ffmpeg.input(name) \
            .output(output_pattern, vf='fps=1') \
            .overwrite_output() \
            .run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("[FFMPEG ERROR]", e.stderr.decode())
        raise RuntimeError("FFmpeg failed to extract frames")

    # Optional: list how many frames were written
    extracted = [
        f for f in os.listdir(frame_dir)
        if f.endswith(".png") and os.path.basename(name) in f
    ]
    print(f"[DEBUG] Extracted {len(extracted)} frame(s).")
    return extracted


def generate_audio_rms_csv(name, csv_filename=None, plot=False):
    y, sr = librosa.load(name, sr=None)
    frame_length = int(sr * 1)  # 100ms frames
    hop_length = frame_length

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    
    if plot:
        plot_time(times, 20 * np.log10(rms), xlabel="Time (s)", ylabel="Volume (dB)", title="Audio Levels Over Time")
    
    if csv_filename is not None:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "RMS"])
            for t, r in zip(times, rms):
                writer.writerow([t, r])


# Save frame brightness to CSV only if a filename is explicitly provided
def generate_frame_brightness_csv(name, csv_filename=None, plot=False, frame_dir="frames"):
    brightness_data = []

    # Use either 1. in-memory return or 2. save to CSV
    is_csv_output = csv_filename is not None

    for fname in sorted(os.listdir(frame_dir)):
        if not fname.endswith(".png"):
            continue

        frame_path = os.path.join(frame_dir, fname)
        print("[DEBUG] Loading frame:", frame_path)

        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARNING] Skipping unreadable frame: {frame_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_data.append(brightness)

    timestamps = [i * (0.1 if is_csv_output else 1) for i in range(len(brightness_data))]

    if is_csv_output:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Brightness"])
            for t, b in zip(timestamps, brightness_data):
                writer.writerow([t, b])

        if plot:
            plot_time(
                timestamps,
                brightness_data,
                xlabel="Time (s)",
                ylabel="Brightness",
                title="Frame Brightness Over Time"
            )
    else:
        return brightness_data, timestamps

def get_frame_paths(frame_dir, ext="png"):
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, f"*.{ext}")))
    return frame_paths

def compute_optical_flow_magnitude(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude)

def process_motion_from_frames(frame_paths, fps=1, verbose=True):
    motion_scores = []
    total_frames = len(frame_paths)

    for i in tqdm(range(total_frames - 1), desc="Processing frames"):
        frame1 = cv2.imread(frame_paths[i])
        frame2 = cv2.imread(frame_paths[i + 1])

        if frame1 is None or frame2 is None:
            if verbose:
                print(f"Warning: Could not read frame {i} or {i+1}, skipping.")
            continue

        motion = compute_optical_flow_magnitude(frame1, frame2)
        timestamp = round(i / fps, 1)
        motion_scores.append([timestamp, motion])

        if verbose:
            print(f"Processed frames {i} and {i+1} at {timestamp}s: motion={motion:.4f}")

    return pd.DataFrame(motion_scores, columns=["Time (s)", "Motion"])

def save_motion_to_csv(motion_df, output_path="motion.csv"):
    motion_df.to_csv(output_path, index=False)
    print(f"Motion data saved to {output_path} with {len(motion_df)} entries.")

def generate_motion_csv(frame_dir="frames", output_path="motion.csv", fps=1):
    frame_paths = get_frame_paths(frame_dir)
    motion_df = process_motion_from_frames(frame_paths, fps=fps)
    save_motion_to_csv(motion_df, output_path)
    return motion_df

def detect_angry_sections(
    input_video,
    csv_filename="angry_sections.csv",
    anger_threshold=0.1,
    merge_gap=7,
    top_n_sections=None,
    whisper_model_size="base",
    verbose=True
):
    # Load Whisper
    model = whisper.load_model(whisper_model_size)
    result = model.transcribe(input_video)

    # Load emotion classifier
    if verbose: print("Loading emotion classifier...")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    emotion_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    labels = ['anger', 'joy', 'optimism', 'sadness']

    angry_segments = []

    for seg in result["segments"]:
        text = seg["text"]
        start = seg["start"]
        end = seg["end"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0]

        anger_score = probs[labels.index("anger")].item()

        if anger_score > anger_threshold:
            angry_segments.append({
                "start": start,
                "end": end,
                "text": text.strip(),
                "anger_score": anger_score
            })

    grouped_angry_sections = []
    if angry_segments:
        current_group = [angry_segments[0]]
        for seg in angry_segments[1:]:
            if seg['start'] - current_group[-1]['end'] <= merge_gap:
                current_group.append(seg)
            else:
                grouped_angry_sections.append(current_group)
                current_group = [seg]
        grouped_angry_sections.append(current_group)

    def sec_to_mmss(sec):
        return str(datetime.timedelta(seconds=int(sec)))

    ranked_sections = []
    for group in grouped_angry_sections:
        start = group[0]['start']
        end = group[-1]['end']
        avg_anger = sum(s['anger_score'] for s in group) / len(group)
        ranked_sections.append({
            "start": start,
            "end": end,
            "avg_anger": avg_anger,
            "segments": group
        })

    ranked_sections.sort(key=lambda x: x["avg_anger"], reverse=True)

    if verbose:
        print("\nAngriest Sections in the Video:\n")
        for i, section in enumerate(ranked_sections if top_n_sections is None else ranked_sections[:top_n_sections]):
            print(f"Section {i+1}")
            print(f"[{sec_to_mmss(section['start'])} → {sec_to_mmss(section['end'])}] | Avg Anger Score: {section['avg_anger']:.2f}\n")
            for seg in section["segments"]:
                print(f"[{sec_to_mmss(seg['start'])} → {sec_to_mmss(seg['end'])}] | Anger: {seg['anger_score']:.2f}")
                print(f"{seg['text']}\n")
            print("------------------------------------------------------------\n")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Section Number",
            "Section Start",
            "Section End",
            "Section Avg Anger",
            "Segment Start",
            "Segment End",
            "Segment Anger Score",
            "Segment Text"
        ])

        for i, section in enumerate(ranked_sections if top_n_sections is None else ranked_sections[:top_n_sections], start=1):
            section_start = sec_to_mmss(section["start"])
            section_end = sec_to_mmss(section["end"])
            avg_anger = f"{section['avg_anger']:.2f}"

            for seg in section["segments"]:
                seg_start = sec_to_mmss(seg["start"])
                seg_end = sec_to_mmss(seg["end"])
                seg_anger = f"{seg['anger_score']:.2f}"
                seg_text = seg["text"].replace('\n', ' ').strip()
                writer.writerow([
                    i,
                    section_start,
                    section_end,
                    avg_anger,
                    seg_start,
                    seg_end,
                    seg_anger,
                    seg_text
                ])

    return ranked_sections


def mmss_to_seconds(time_str):
    t = datetime.datetime.strptime(time_str, "%H:%M:%S")
    return t.minute * 60 + t.second

def interpolate_numeric_field(data, key):
    times = [entry["time"] for entry in data]
    values = [entry[key] if isinstance(entry[key], float) else np.nan for entry in data]
    interpolated = np.interp(
        times,
        [t for t, v in zip(times, values) if not np.isnan(v)],
        [v for v in values if not np.isnan(v)]
    )
    for i, val in enumerate(values):
        if np.isnan(val):
            data[i][key] = round(interpolated[i], 6)

def merge_features_to_csv(
    anger_csv="angry_sections.csv",
    motion_csv="motion.csv",
    rms_csv="audio_rms.csv",
    brightness_csv="frame_brightness.csv",
    output_csv="normalized_merged_data.csv",
    time_step=0.1,
    interpolate_anger=True,
    verbose=True
):
    angry_segments = []
    with open(anger_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            angry_segments.append({
                "section": int(row["Section Number"]),
                "start": mmss_to_seconds(row["Segment Start"]),
                "end": mmss_to_seconds(row["Segment End"]),
                "text": row["Segment Text"],
                "anger": float(row["Segment Anger Score"]),
            })

    def load_scalar_csv(path, key_name):
        data = {}
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = float(row["Time (s)"])
                data[round(time, 1)] = float(row[key_name])
        return data

    motion_data = load_scalar_csv(motion_csv, "Motion")
    rms_data = load_scalar_csv(rms_csv, "RMS")
    brightness_data = load_scalar_csv(brightness_csv, "Brightness")

    min_time = 0.0
    max_time = max(
        max(rms_data.keys(), default=0),
        max(brightness_data.keys(), default=0),
        max((seg["end"] for seg in angry_segments), default=0)
    )

    all_times = np.round(np.arange(min_time, max_time + time_step, time_step), 1)

    merged = []
    for t in all_times:
        section = ""
        text = ""
        anger = "n/a"

        for seg in angry_segments:
            if seg["start"] <= t <= seg["end"]:
                section = seg["section"]
                text = seg["text"]
                anger = seg["anger"]
                break

        rms = rms_data.get(t, "n/a")
        brightness = brightness_data.get(t, "n/a")
        motion = motion_data.get(t, "n/a")

        merged.append({
            "time": t,
            "section": section,
            "text": text,
            "anger": anger,
            "rms": rms,
            "brightness": brightness,
            "motion": motion
        })

    interpolate_numeric_field(merged, "rms")
    interpolate_numeric_field(merged, "brightness")
    interpolate_numeric_field(merged, "motion")
    if interpolate_anger:
        interpolate_numeric_field(merged, "anger")

    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Section", "Text", "Anger Score", "RMS", "Brightness", "Motion"])
        for row in merged:
            writer.writerow([
                f"{row['time']:.1f}",
                row["section"],
                row["text"],
                f"{row['anger']:.4f}" if isinstance(row["anger"], float) else row["anger"],
                f"{row['rms']:.6f}" if isinstance(row["rms"], float) else row["rms"],
                f"{row['brightness']:.6f}" if isinstance(row["brightness"], float) else row["brightness"],
                f"{row['motion']:.6f}" if isinstance(row["motion"], float) else row["motion"],
            ])

    if verbose:
        print(f"Saved merged and interpolated features to {output_csv}")

def filter_spaced_timestamps(df, time_col="Time (s)", min_spacing=10):
    spaced_times = []
    last_time = -float('inf')
    for t in sorted(df[time_col]):
        if t - last_time > min_spacing + 1:  # Strictly greater than min_spacing
            spaced_times.append(t)
            last_time = t
    return spaced_times