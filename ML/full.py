import os
import joblib
import pandas as pd
from fullflight2 import get_video, detect_angry_sections, generate_motion_csv, merge_features_to_csv, generate_audio_rms_csv, generate_frame_brightness_csv, extract_frames, filter_spaced_timestamps


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
    # extract_frames(name=input_video_path)
    
    # 1) Detect angry sections & write angry_sections.csv
    # detect_angry_sections(
    #     input_video=input_video_path,
    #     csv_filename=angry_csv,
    #     anger_threshold=anger_threshold,
    #     merge_gap=merge_gap,
    #     top_n_sections=top_n_angry_sections,
    #     whisper_model_size=whisper_model_size,
    #     verbose=verbose
    # )
    
    

    # 2) Extract frames (if not done) and compute motion.csv
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        # You need to extract frames from video first (ffmpeg or cv2)
        # Example: os.system(f"ffmpeg -i {input_video_path} -vf fps={fps} {frames_dir}/frame_%04d.png")
        
        raise RuntimeError(f"Frames folder '{frames_dir}' is empty, extract frames before running motion extraction")

    # generate_motion_csv(frame_dir=frames_dir, output_path=motion_csv, fps=fps)
    # generate_audio_rms_csv(input_video=input_video_path, csv_filename=rms_csv, plot=False)
    # generate_frame_brightness_csv(name=frames_dir, csv_filename=brightness_csv, plot=False)

    # 3) Assume audio RMS and brightness CSVs are generated externally or add those steps here

    # 4) Merge all features into merged CSV
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

    # 5) Load model and predict fight scenes
    clf = joblib.load(model_path)
    new_df = pd.read_csv(merged_csv)

    new_df = new_df[
        (new_df["Anger Score"] != 'n/a') & 
        (new_df["RMS"] != 'n/a') & 
        (new_df["Brightness"] != 'n/a') & 
        (new_df["Motion"] != 'n/a')
    ]

    new_df["Anger Score"] = new_df["Anger Score"].astype(float)
    new_df["RMS"] = new_df["RMS"].astype(float)
    new_df["Brightness"] = new_df["Brightness"].astype(float)
    new_df["Motion"] = new_df["Motion"].astype(float)

    X_new = new_df[["Anger Score", "RMS", "Brightness", "Motion"]]
    new_df["Fight Probability"] = clf.predict_proba(X_new)[:, 1]

    fight_df = new_df[new_df["Fight Probability"] > fight_prob_threshold]

    if verbose:
        print(f"Detected {len(fight_df)} probable fight scene frames above threshold {fight_prob_threshold}")
        
    results = filter_spaced_timestamps(fight_df, time_col="Time (s)", min_spacing=10)

    return results

results = full_fight_scene_pipeline("uploads/[Kayoanime] Solo Leveling - S02E06 (1).mkv")

print("Filtered timestamps spaced more than 10 seconds apart:")
for t in results:
    print(f"{t:.1f}")