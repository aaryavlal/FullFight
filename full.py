import os
import joblib
import pandas as pd
from fullflight2 import (
    get_video,
    detect_angry_sections,
    generate_motion_csv,
    merge_features_to_csv,
    generate_audio_rms_csv,
    generate_frame_brightness_csv,
    extract_frames,
    filter_spaced_timestamps
)

def full_fight_scene_pipeline(
    input_video_path,
    model_path="rf_fight_scene_model.pkl",
    frames_dir="frames",
    motion_csv="motion.csv",
    rms_csv="audio_rms.csv",
    brightness_csv="frame_brightness.csv",
    angry_csv="angry_sections.csv",
    merged_csv="normalized_merged_data.csv",
    anger_threshold=0.1,
    merge_gap=7,
    top_n_angry_sections=5,
    whisper_model_size="base",
    fps=1,
    time_step=0.1,
    fight_prob_threshold=0.8,
    verbose=True
):
    """
    Run the full ML pipeline to extract likely fight scene timestamps from a video.
    Returns: List of timestamp floats (in seconds).
    """

    # Step 1: Check if frames are extracted — raise error if missing
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        raise RuntimeError(
            f"Frames folder '{frames_dir}' is empty. "
            f"Extract frames before running this pipeline."
        )

    # Step 2: [Optional] Generate motion/audio/brightness CSVs if needed
    # generate_motion_csv(frame_dir=frames_dir, output_path=motion_csv, fps=fps)
    # generate_audio_rms_csv(input_video=input_video_path, csv_filename=rms_csv, plot=False)
    # generate_frame_brightness_csv(name=frames_dir, csv_filename=brightness_csv, plot=False)

    # Step 3: Merge all features to a single CSV
    merge_features_to_csv(
        anger_csv=angry_csv,
        motion_csv=motion_csv,
        rms_csv=rms_csv,
        brightness_csv=brightness_csv,
        output_csv=merged_csv,
        time_step=time_step,
        interpolate_anger=True,
        verbose=verbose
    )

    # Step 4: Run the classifier on merged features
    clf = joblib.load(model_path)
    df = pd.read_csv(merged_csv)

    # Clean out incomplete rows
    df = df[
        (df["Anger Score"] != 'n/a') &
        (df["RMS"] != 'n/a') &
        (df["Brightness"] != 'n/a') &
        (df["Motion"] != 'n/a')
    ]

    # Convert columns to float
    df["Anger Score"] = df["Anger Score"].astype(float)
    df["RMS"] = df["RMS"].astype(float)
    df["Brightness"] = df["Brightness"].astype(float)
    df["Motion"] = df["Motion"].astype(float)

    # Make predictions
    features = df[["Anger Score", "RMS", "Brightness", "Motion"]]
    df["Fight Probability"] = clf.predict_proba(features)[:, 1]

    # Filter for high-probability fight scenes
    fight_df = df[df["Fight Probability"] > fight_prob_threshold]

    if verbose:
        print(f"🧠 Detected {len(fight_df)} probable fight-scene frames above threshold {fight_prob_threshold}")

    # Step 5: Space out timestamps (e.g. no closer than 10s)
    results = filter_spaced_timestamps(fight_df, time_col="Time (s)", min_spacing=10)

    return results
