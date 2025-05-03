import os
import numpy as np
import pandas as pd
import hdf5_getters as h5
from pathlib import Path
from tqdm import tqdm

def extract_features_from_h5(h5_file_path):
    f = h5.open_h5_file_read(h5_file_path)
    try:
        track_id = h5.get_track_id(f).decode("utf-8")
        features = {
            "track_id": track_id,
            "tempo": h5.get_tempo(f),
            "loudness": h5.get_loudness(f),
            "duration": h5.get_duration(f),
            "key": h5.get_key(f),
            "mode": h5.get_mode(f),
            "time_signature": h5.get_time_signature(f),
        }
    except Exception as e:
        features = None
    f.close()
    return features

def load_all_features(summary_folder):
    all_features = []
    for h5_file in tqdm(Path(summary_folder).rglob("*.h5"), desc="Extracting audio features"):
        feat = extract_features_from_h5(str(h5_file))
        if feat:
            all_features.append(feat)
    return pd.DataFrame(all_features)
