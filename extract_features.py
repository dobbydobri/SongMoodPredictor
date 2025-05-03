from pathlib import Path
import pandas as pd
import hdf5_getters as h5
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
    except Exception:
        features = None
    f.close()
    return features

def load_all_features(folder_path):
    all_features = []
    for h5_file in tqdm(Path(folder_path).rglob("*.h5"), desc="Extracting audio features"):
        features = extract_features_from_h5(str(h5_file))
        if features:
            all_features.append(features)
    return pd.DataFrame(all_features)

if __name__ == "__main__":
    print("Extracting features...")
    df = load_all_features("data/raw/MillionSongSubset")
    df.to_csv("features.csv", index=False)
    print(f"Saved {len(df)} rows to features.csv")
