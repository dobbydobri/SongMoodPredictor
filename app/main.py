import nest_asyncio
import streamlit as st
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

nest_asyncio.apply()

st.title("🎵 Mood Predictor")
st.write("This app will predict the mood of a song based on audio features.")

# Load songs metadata CSV
songs_df = pd.read_csv('../data/raw/features/metadata/all_songs.csv')

excluded_song_ids = ['48','3','116','386','47','324','646','691','634','329','656',
                     '622','152','637','8','174','769','149','278','54','488','996','113','1390','2006']

test_songs_df = songs_df[songs_df['song_id'].astype(str).isin(excluded_song_ids)]

test_song_options = test_songs_df.apply(lambda row: f"{row['title']} — {row['artist']}", axis=1).tolist()
song_to_id = dict(zip(test_song_options, test_songs_df['song_id'].astype(str)))

# Load merged features and label encoder
merged_df = pd.read_csv('../models/merged_features.csv')
le = LabelEncoder()
le.classes_ = np.load('../models/le_classes.npy', allow_pickle=True)

merged_df['track_id'] = merged_df['track_id'].astype(str).str.strip()

exclude_cols = ['track_id', 'mood', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std', 'valence_norm', 'arousal_norm']
feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
features_dict = dict(zip(merged_df['track_id'], merged_df[feature_cols].values))

scaler = joblib.load('../models/scaler.save')
edge_index = torch.load('../models/edge_index.pt')
sorted_track_ids = np.load('../models/sorted_track_ids.npy').tolist()

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

all_features = np.array([features_dict[tid] for tid in sorted_track_ids])
all_features_scaled = scaler.transform(all_features)
all_features_tensor = torch.tensor(all_features_scaled, dtype=torch.float).to(device)
graph_data = torch_geometric.data.Data(x=all_features_tensor, edge_index=edge_index.to(device))

if 'mood' not in merged_df.columns:
    raise ValueError("The 'mood' column is missing from the DataFrame")

labels_encoded = le.transform(merged_df['mood'])
graph_data.y = torch.tensor(labels_encoded, dtype=torch.long).to(device)

def predict_mood(track_id: str):
    if track_id not in sorted_track_ids:
        return "Unknown (features not found)", None
    node_idx = sorted_track_ids.index(track_id)
    with torch.no_grad():
        out = model(graph_data)
        pred_class = out[node_idx].argmax(dim=0).item()
        pred_mood = le.inverse_transform([pred_class])[0]
    return pred_mood, node_idx

# Add session state for mood image and statistics toggles
if 'show_mood_image' not in st.session_state:
    st.session_state.show_mood_image = False
if 'show_statistics' not in st.session_state:
    st.session_state.show_statistics = False

# Layout for buttons
colA, colB = st.columns([1, 1])

with colA:
    mood_dist_button = st.button("Show Mood Distribution")
with colB:
    stats_button = st.button("Show Statistics")

# Toggle mood image visibility
if mood_dist_button:
    st.session_state.show_mood_image = not st.session_state.show_mood_image

# Toggle statistics visibility
if stats_button:
    st.session_state.show_statistics = not st.session_state.show_statistics

if st.session_state.show_mood_image:
    st.image('../mood_distribution_graph.png', use_container_width=True)

if st.session_state.show_statistics:
    G_temp = nx.Graph()
    G_temp.add_edges_from(edge_index.t().cpu().numpy())

    st.subheader("Graph Statistics")
    st.write(f"**Total Nodes:** {G_temp.number_of_nodes()}")
    st.write(f"**Total Edges:** {G_temp.number_of_edges()}")

    mood_counts = merged_df['mood'].value_counts().to_dict()
    st.write("**Nodes per Mood:**")
    for mood in ['happy', 'sad', 'calm', 'angry']:
        st.write(f"- {mood.capitalize()}: {mood_counts.get(mood, 0)}")

def plot_graph_with_highlighted_node(node_idx, edge_index, predicted_mood, data):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().cpu().numpy())
    neighbors = list(G.neighbors(node_idx))
    mood_colors = {'happy': 'yellow', 'sad': 'blue', 'angry': 'red', 'calm': 'green'}

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [200] * len(G.nodes)
    node_colors = ['lightblue'] * len(G.nodes)

    for i in range(len(G.nodes)):
        if i != node_idx and i not in neighbors:
            node_sizes[i] = 50

    node_sizes[node_idx] = 450
    node_colors[node_idx] = 'black'

    for neighbor in neighbors:
        mood = le.inverse_transform([data.y[neighbor].item()])[0]
        node_colors[neighbor] = mood_colors.get(mood, 'grey')

    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes,
            font_size=8, font_weight='bold', edge_color='grey', alpha=0.7, width=0.5)

    patches = [mpatches.Patch(color=color, label=label) for label, color in mood_colors.items()]
    plt.legend(handles=patches, loc='upper right')
    plt.title(f"Predicted Mood (black node): {predicted_mood}", fontsize=14)
    st.pyplot(plt)

song_choice = st.selectbox("Choose a song from the test set:", test_song_options)
col1, col2 = st.columns([2, 1])

with col1:
    predict_button = st.button("Predict Mood")
with col2:
    play_button = st.button("Play Song")

song_file = f'../data/raw/DEAM_audio/MEMD_audio/{song_to_id[song_choice]}.mp3'

if play_button:
    st.audio(song_file)

plot_placeholder = st.empty()

if predict_button:
    plot_placeholder.text("Plotting the graph live...")
    track_id = song_to_id[song_choice]
    predicted_mood, node_idx = predict_mood(track_id)
    st.success(f"The predicted mood for '{song_choice}' is: {predicted_mood}")
    plot_graph_with_highlighted_node(node_idx, edge_index, predicted_mood, graph_data)
    plot_placeholder.empty()
