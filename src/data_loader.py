import json
from pathlib import Path
from collections import defaultdict

# List of keywords to search for in tags (substring match)
MOOD_KEYWORDS = [
    "adrenaline", "aggress", "ambient", "angry", "bitter", "blue", "calm", "cheer", "chill",
    "comforting", "cry", "depress", "despair", "down", "drama", "dramatic", "drive",
    "ecstatic", "emotional", "energetic", "euphoric", "expressive", "fast", "feelgood", "fun",
    "fury", "groove", "happy", "heartache", "heartbreak", "heartfelt", "hopeful", "hype",
    "inspirational", "intense", "joy", "joyous", "lonely", "longing", "meditat", "melancholy",
    "mellow", "melodic", "nostalgic", "optimistic", "passion", "peace", "power", "pump",
    "quiet", "rage", "relax", "romantic", "sad", "sentimental", "serene", "smile", "soothing",
    "tear", "triumphant", "uplift", "vengeful", "vibe", "vulnerable", "yearning"
]

TAG_TO_MOOD = {
    "happy": [
        "happy", "cheer", "smile", "fun", "feelgood", "uplift", "joy", "joyous", "optimistic", "hopeful", "ecstatic", "euphoric"
    ],
    "sad": [
        "sad", "cry", "tear", "blue", "down", "depress", "melancholy", "heartache", "heartbreak", "lonely", "bitter", "despair", "nostalgic", "yearning"
    ],
    "calm": [
        "calm", "chill", "relax", "peace", "soothing", "ambient", "serene", "quiet", "mellow", "comforting", "meditat", "melodic"
    ],
    "energetic": [
        "energetic", "fast", "drive", "adrenaline", "pump", "hype", "groove", "power", "triumphant"
    ],
    "emotional": [
        "emotional", "expressive", "passion", "sentimental", "dramatic", "heartfelt", "romantic", "vulnerable", "longing", "inspirational"
    ],
    "angry": [
        "angry", "rage", "aggress", "fury", "vengeful", "intense"
    ]
}

def load_track_json_files(folder_path):
    """Load all JSON track files from a folder into a list of dicts."""
    tracks = []
    for file in Path(folder_path).rglob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                track = json.load(f)
                tracks.append(track)
        except json.JSONDecodeError:
            continue  # skip broken files
    return tracks

def tag_matches_mood(tag_str, mood_keywords=MOOD_KEYWORDS):
    """Check if any mood keyword is in a tag (substring match)."""
    return any(mood in tag_str.lower() for mood in mood_keywords)

def extract_mood_labeled_tracks(tracks, mood_keywords=MOOD_KEYWORDS):
    """Filter tracks that contain any mood-related tags."""
    mood_tracks = []
    for t in tracks:
        matched_tags = [
            tag[0] for tag in t.get("tags", [])
            if tag_matches_mood(tag[0], mood_keywords)
        ]
        if matched_tags:
            mood_tracks.append({
                "track_id": t.get("track_id"),
                "title": t.get("title"),
                "artist": t.get("artist"),
                "tags": matched_tags
            })
    return mood_tracks


def map_tags_to_mood(tags):
    """Map a list of tags to one mood label using TAG_TO_MOOD."""
    mood_counts = defaultdict(int)
    for tag in tags:
        for mood, keywords in TAG_TO_MOOD.items():
            if any(k in tag.lower() for k in keywords):
                mood_counts[mood] += 1
    if not mood_counts:
        return None
    # Return the most matched mood
    return max(mood_counts, key=mood_counts.get)