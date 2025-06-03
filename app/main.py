import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

st.title("🎵 Mood Predictor")
st.write("This app will predict the mood of a song based on audio features.")

# Load songs metadata CSV
songs_df = pd.read_csv('../data/raw/features/metadata/all_songs.csv')

excluded_song_ids = ['48','3','116','386','47','324','646','691','634','329','656',
                     '622','152','637','8','174','769','149','278','54','488','996','113']

test_songs_df = songs_df[songs_df['song_id'].astype(str).isin(excluded_song_ids)]

test_song_options = test_songs_df.apply(lambda row: f"{row['title']} — {row['artist']}", axis=1).tolist()
song_to_id = dict(zip(test_song_options, test_songs_df['song_id'].astype(str)))

# Load merged features and label encoder
merged_df = pd.read_csv('../models/merged_features.csv')
le = LabelEncoder()
le.classes_ = np.load('../models/le_classes.npy', allow_pickle=True)

merged_df['track_id'] = merged_df['track_id'].astype(str).str.strip()
feature_cols = [col for col in merged_df.columns if col != 'track_id']

features_dict = dict(zip(merged_df['track_id'], merged_df[feature_cols].values))

# Load scaler, edge_index, and sorted_track_ids for consistent ordering
scaler = joblib.load('../models/scaler.save')
edge_index = torch.load('../models/edge_index.pt')

sorted_track_ids = np.load('../models/sorted_track_ids.npy').tolist()  # <-- Load saved order here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = len(feature_cols)
hidden_dim = 32
output_dim = len(le.classes_)

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.7):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load('../models/graphsage_model.pth', map_location=device))
model.eval()

# Build features tensor in the exact order of sorted_track_ids
all_features = np.array([features_dict[tid] for tid in sorted_track_ids])
all_features_scaled = scaler.transform(all_features)
all_features_tensor = torch.tensor(all_features_scaled, dtype=torch.float).to(device)

graph_data = torch_geometric.data.Data(x=all_features_tensor, edge_index=edge_index.to(device))

def predict_mood(track_id: str):
    if track_id not in sorted_track_ids:
        return "Unknown (features not found)"
    node_idx = sorted_track_ids.index(track_id)

    with torch.no_grad():
        out = model(graph_data)
        logits = out[node_idx].cpu().numpy()
        pred_class = out[node_idx].argmax(dim=0).item()
        pred_mood = le.inverse_transform([pred_class])[0]

    return pred_mood

song_choice = st.selectbox("Choose a song:", test_song_options)

if st.button("Predict Mood"):
    track_id = song_to_id[song_choice]
    predicted_mood = predict_mood(track_id)
    st.success(f"The predicted mood for '{song_choice}' is: {predicted_mood}")
