import streamlit as st
import pandas as pd
from scipy import sparse
import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender import load_data, recommend_track

# --- Page config ---
st.set_page_config(page_title="Harmony.ai Music Recommender", layout="wide")

# --- Load data ---
@st.cache_data
def load_all():
    df, X = load_data()
    return df, X

df, X = load_all()

# --- App UI ---
st.title("🎵 Harmony.ai Music Recommender")
st.write("Get song recommendations based on your favorite track!")

track_name = st.text_input("Enter a song name:")

top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=10)

if st.button("Recommend"):
    if track_name.strip() == "":
        st.warning("Please enter a song name.")
    else:
        recs = recommend_track(track_name, df, X, top_n=top_n)
        if recs is None or len(recs) == 0:
            st.info("No recommendations found for this song.")
        else:
            st.success(f"Top {len(recs)} recommendations for '{track_name}':")
            # Clean up the artist names (convert list to string)
            recs['artists'] = recs['artists'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else ', '.join(x))
            # Reset index to start from 1
            recs.index = range(1, len(recs) + 1)
            st.dataframe(recs)
