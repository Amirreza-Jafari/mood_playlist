import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mood categories and their descriptions
MOOD_DESCRIPTIONS = {
    "Ultra Hype": "Maximum energy, perfect for parties and high-intensity moments!",
    "Hype": "High-energy tracks to get you motivated and excited.",
    "Energetic": "Upbeat songs to keep your spirits high.",
    "Moderate": "Balanced tracks for a steady, pleasant mood.",
    "Chill": "Relaxed vibes to help you unwind.",
    "Mellow": "Soft, low-key music for quiet moments."
}

@st.cache_resource
def load_and_preprocess_data():
    """
    Load and preprocess dataset with enhanced error handling
    """
    try:
        # Load dataset
        df = pd.read_csv("train.csv")
        st.write(f"Dataset loaded. Total tracks: {len(df)}")

        features = ["tempo", "energy", "danceability"]
        
        # Robust preprocessing
        df = df.dropna(subset=features)
        df = df.drop_duplicates(subset=["track_name", "artists"])

        # Advanced feature scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df[features])
        scaled_cols = [f"{feat}_scaled" for feat in features]
        df_scaled = pd.DataFrame(X_scaled, columns=scaled_cols, index=df.index)
        df = pd.concat([df, df_scaled], axis=1)

        # Compute mood score with dynamic weighting
        def compute_mood_score(row, 
            weights={'energy': 0.4, 'tempo': 0.3, 'danceability': 0.3}):
            return sum(weights[feat] * row[f'{feat}_scaled'] for feat in weights)

        df['mood_score'] = df.apply(compute_mood_score, axis=1)

        # More nuanced mood categorization
        mood_labels = ["Ultra Hype", "Hype", "Energetic", "Moderate", "Chill", "Mellow"]
        df['mood_category'] = pd.qcut(df['mood_score'], 
                                       q=len(mood_labels), 
                                       labels=mood_labels)

        return df

    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

def get_playlist_by_mood(df, mood_query: str, playlist_size: int = 5):
    """
    Enhanced playlist generation
    """
    if df is None:
        st.error("Dataset not loaded")
        return pd.DataFrame()

    # Find exact or closest mood match
    mood_categories = df['mood_category'].unique()
    exact_match = [m for m in mood_categories if m.lower() == mood_query.lower()]
    
    mood_category = exact_match[0] if exact_match else mood_categories[0]

    # Filter and sample playlist
    filtered = df[df['mood_category'] == mood_category]
    playlist = filtered.sample(n=min(playlist_size, len(filtered)))
    
    # Add Spotify links
    playlist['spotify_link'] = playlist['track_id'].apply(
        lambda tid: f"https://open.spotify.com/track/{tid}" 
        if pd.notnull(tid) else "Link unavailable"
    )

    return playlist

def main():
    st.set_page_config(
        page_title="Mood Playlist Generator",
        page_icon="🎵",
        layout="wide"
    )

    # Load data
    df = load_and_preprocess_data()
    
    st.title("🎵 Mood Playlist Generator")
    
    # Sidebar for mood selection
    st.sidebar.header("Playlist Customization")
    
    # Mood selection with descriptions
    mood_options = ["Ultra Hype", "Hype", "Energetic", "Moderate", "Chill", "Mellow"]
    selected_mood = st.sidebar.selectbox(
        "Choose Your Mood", 
        mood_options, 
        format_func=lambda x: f"{x} - {MOOD_DESCRIPTIONS[x]}"
    )
    
    # Playlist size with smart slider
    playlist_size = st.sidebar.slider(
        "Number of Songs", 
        min_value=3, 
        max_value=10, 
        value=5,
        help="Select how many songs you want in your playlist"
    )
    
    # Generate Playlist
    if st.sidebar.button("Generate Playlist"):
        if df is not None:
            # Get playlist
            playlist = get_playlist_by_mood(df, selected_mood, playlist_size)
            
            # Display mood description
            st.subheader(f"{selected_mood} Mood Playlist")
            st.write(MOOD_DESCRIPTIONS[selected_mood])
            
            # Create columns for playlist display
            cols = st.columns(3)
            
            for i, (_, song) in enumerate(playlist.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                    ### {song['track_name']}
                    #### {song['artists']}
                    - **Tempo:** {song['tempo']:.2f}
                    - **Energy:** {song['energy']:.2f}
                    - **Danceability:** {song['danceability']:.2f}
                    [Listen on Spotify]({song['spotify_link']})
                    """)
        else:
            st.error("Failed to load dataset. Please check the data source.")

if __name__ == "__main__":
    main()
