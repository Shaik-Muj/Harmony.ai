import pandas as pd
import numpy as np
import os

def load_and_clean_data(raw_path="data/raw/tracks.csv", save_path="data/processed"):
    """
    Loads Spotify dataset, cleans missing values, and saves a processed version.
    """
    # Loading into memory
    print("=====   Loading dataset...   =====")
    df = pd.read_csv(raw_path)
    print(f"=====   Loaded {len(df):,} tracks   =====")

    # Remove duplicates
    df.drop_duplicates(subset="id", inplace=True)

    # Handle missing values (fill numeric with median, text with mode)
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Convert release_date to datetime (ignore errors)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # Drop rows without valid release dates
    df = df.dropna(subset=["release_date"])

    # Keep only relevant columns for recommendations
    feature_cols = [
        "id", "name", "artists", "id_artists", "popularity", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms"
    ]
    df = df[feature_cols]

    # Normalize numeric features (for cosine similarity later)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[["danceability", "energy", "loudness", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]] = \
        scaler.fit_transform(df_scaled[["danceability", "energy", "loudness", "speechiness", "acousticness",
                                        "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]])

    # Save processed data
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "tracks_processed.csv")
    df_scaled.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset saved to {output_file}")
    return df_scaled

def load_and_clean_artists(raw_path="data/raw/artists.csv", save_path="data/processed"):
    """
    Loads Spotify artists dataset, cleans missing values, and saves a processed version.
    """
    print("=====   Loading artists dataset...   =====")
    df = pd.read_csv(raw_path)
    print(f"✅ Loaded {len(df):,} artists")

    # Remove duplicates
    df.drop_duplicates(subset="id", inplace=True)

    # Handle missing values
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Normalize numeric features if they exist
    from sklearn.preprocessing import MinMaxScaler
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Keep relevant columns for recommendation
    # Adjust based on your dataset columns
    feature_cols = ["id", "name", "followers", "popularity", "genres"]
    df = df[[col for col in feature_cols if col in df.columns]]

    # Save processed artists data
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "artists_processed.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned artists dataset saved to {output_file}")
    return df


if __name__ == "__main__":
    df_tracks = load_and_clean_data()
    df_artists = load_and_clean_artists()
    print("✅ Data preprocessing completed successfully.")
