import pandas as pd
import os

# Define the input files
file_paths = [
    '../data/raw/features/metadata/metadata_2013.csv',
    '../data/raw/features/metadata/metadata_2014.csv',
    '../data/raw/features/metadata/metadata_2015.csv'
]

def find_column(columns, possible_names):
    """Return the first matching column name from possible_names found in columns, else None"""
    for name in possible_names:
        if name in columns:
            return name
    return None

# Read and combine all CSV files
dataframes = []
for path in file_paths:
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        
        # Detect the actual column names for song id, title, artist
        song_id_col = find_column(df.columns, ['song_id', 'Id', 'id'])
        title_col = find_column(df.columns, ['Song title', 'title', 'Track'])
        artist_col = find_column(df.columns, ['Artist', 'artist'])

        if not (song_id_col and title_col and artist_col):
            print(f"Warning: Could not find all necessary columns in {path}")
            continue

        # Rename columns to standardized names
        df = df.rename(columns={
            song_id_col: 'song_id',
            title_col: 'title',
            artist_col: 'artist'
        })

        # Select only these columns
        df = df[['song_id', 'title', 'artist']]

        dataframes.append(df)
    except Exception as e:
        print(f"Failed to read {path}: {e}")

# Combine all dataframes
all_data = pd.concat(dataframes, ignore_index=True)

# Clean text columns
all_data['song_id'] = all_data['song_id'].astype(str).str.strip()
all_data['title'] = all_data['title'].astype(str).str.replace(r'\t', '', regex=True).str.strip()
all_data['artist'] = all_data['artist'].astype(str).str.replace(r'\t', '', regex=True).str.strip()

# Drop rows with missing critical info
all_data = all_data.dropna(subset=['song_id', 'title', 'artist'])

# Save to CSV
output_path = '../data/raw/features/metadata/all_songs.csv'
all_data.to_csv(output_path, index=False)
print(f"Saved cleaned song metadata to {output_path}")
