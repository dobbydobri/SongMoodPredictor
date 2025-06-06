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

# Exclude mood, valence, and arousal columns from the feature set
exclude_cols = ['track_id', 'mood', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std', 
                'valence_norm', 'arousal_norm']  # Exclude mood and arousal/valence columns from features

feature_cols = [col for col in merged_df.columns if col not in exclude_cols]  # Get the remaining features
features_dict = dict(zip(merged_df['track_id'], merged_df[feature_cols].values))

# Load scaler, edge_index, and sorted_track_ids for consistent ordering
scaler = joblib.load('../models/scaler.save')
edge_index = torch.load('../models/edge_index.pt')

sorted_track_ids = np.load('../models/sorted_track_ids.npy').tolist()  # <-- Load saved order here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure input_dim is correct based on the feature columns
input_dim = len(feature_cols)  # Exclude mood-related columns

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

# Instantiate and load the model
model = GraphSAGE(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load('../models/graphsage_model.pth', map_location=device))  # Load pre-trained weights
model.eval()

# Build features tensor in the exact order of sorted_track_ids
all_features = np.array([features_dict[tid] for tid in sorted_track_ids])
all_features_scaled = scaler.transform(all_features)
all_features_tensor = torch.tensor(all_features_scaled, dtype=torch.float).to(device)

# Creating graph_data for the model
graph_data = torch_geometric.data.Data(x=all_features_tensor, edge_index=edge_index.to(device))

# Ensure 'mood' column exists and assign labels to graph_data
if 'mood' not in merged_df.columns:
    raise ValueError("The 'mood' column is missing from the DataFrame")

# Encode the mood labels and assign them to graph_data.y
labels_encoded = le.transform(merged_df['mood'])
graph_data.y = torch.tensor(labels_encoded, dtype=torch.long).to(device)  # Assign the labels to graph_data

def predict_mood(track_id: str):
    if track_id not in sorted_track_ids:
        return "Unknown (features not found)"
    node_idx = sorted_track_ids.index(track_id)

    with torch.no_grad():
        out = model(graph_data)
        logits = out[node_idx].cpu().numpy()
        pred_class = out[node_idx].argmax(dim=0).item()
        pred_mood = le.inverse_transform([pred_class])[0]

    return pred_mood, node_idx  # Return the mood and node index for graph plotting

# **New Section: Button for displaying the image**
# This button will appear above the song selection and toggle the image visibility
if 'show_mood_image' not in st.session_state:
    st.session_state.show_mood_image = False

mood_dist_button = st.button("Show Mood Distribution")

if mood_dist_button:
    # Toggle the state of the image visibility
    st.session_state.show_mood_image = not st.session_state.show_mood_image

# Show the mood distribution image if the button has been pressed
if st.session_state.show_mood_image:
    st.image('../mood_distribution_graph.png', use_container_width=True)

def plot_graph_with_highlighted_node(node_idx, edge_index, predicted_mood, data):
    # Create a graph using NetworkX
    G = nx.Graph()
    
    # Add nodes and edges to the graph
    G.add_edges_from(edge_index.t().cpu().numpy())
    
    # Get the neighbors of the predicted node
    neighbors = list(G.neighbors(node_idx))
    
    # Assign colors based on moods
    mood_colors = {
        'happy': 'yellow',
        'sad': 'blue',
        'angry': 'red',
        'calm': 'green',
    }

    # Plot the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
    node_sizes = [200] * len(G.nodes)  # Default size for nodes
    
    # Make irrelevant nodes smaller and almost invisible
    for i in range(len(G.nodes)):
        if i != node_idx and i not in neighbors:
            node_sizes[i] = 50  # Reduce size of irrelevant nodes
    
    node_colors = ['lightblue'] * len(G.nodes)
    
    # Highlight the predicted node (make it black)
    node_sizes[node_idx] = 450  # Increase the size of the predicted node (1.5x)
    node_colors[node_idx] = 'black'  # Always color the predicted node with black
    
    # Highlight neighbors of the predicted node based on their actual mood
    for neighbor in neighbors:
        neighbor_mood = le.inverse_transform([data.y[neighbor].item()])[0]  # Get the mood of the neighbor
        node_colors[neighbor] = mood_colors.get(neighbor_mood, 'gray')  # Assign color based on neighbor's mood
    
    # Draw the nodes and edges with adjusted sizes and colors
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, font_size=8, font_weight='bold', edge_color='gray', alpha=0.7)
    
    # Add a legend for moods
    patches = [mpatches.Patch(color=color, label=label) for label, color in mood_colors.items()]
    plt.legend(handles=patches, loc='upper right')

    # Add a title with the predicted mood
    plt.title(f"Predicted Mood: {predicted_mood}", fontsize=14)
    st.pyplot(plt)  # Use Streamlit's method to display the plot

# Streamlit UI for song selection
song_choice = st.selectbox("Choose a song from the test set:", test_song_options)

# Streamlit layout for buttons: predict button left, play button right
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction button (left)
    predict_button = st.button("Predict Mood")

with col2:
    # Play song button (right)
    play_button = st.button("Play Song")

# Get the corresponding file path for the selected song based on song_id
song_file = f'../data/raw/DEAM_audio/MEMD_audio/{song_to_id[song_choice]}.mp3'

# Display the audio player if a song is selected
if play_button:
    st.audio(song_file)

# Show a placeholder while the graph is being plotted
plot_placeholder = st.empty()

# Prediction and graph plotting when button is clicked
if predict_button:
    # Show message while plotting
    plot_placeholder.text("Plotting the graph...")

    track_id = song_to_id[song_choice]
    predicted_mood, node_idx = predict_mood(track_id)  # Get both mood and node index
    st.success(f"The predicted mood for '{song_choice}' is: {predicted_mood}")
    
    # Plot the graph highlighting the node and its neighbors
    plot_graph_with_highlighted_node(node_idx, edge_index, predicted_mood, graph_data)

    # Clear the placeholder message after plotting
    plot_placeholder.empty()
