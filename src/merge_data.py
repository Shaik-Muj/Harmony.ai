import pandas as pd
import os

def merge_tracks_artists(tracks_path="data/processed/tracks_processed.csv",
                         artists_path="data/processed/artists_processed.csv",
                         save_path="data/processed/tracks_artists_merged.csv"):
    # Load processed CSVs
    tracks_df = pd.read_csv(tracks_path)
    artists_df = pd.read_csv(artists_path)

    # The join key depends on your CSV columns
    merged_df = pd.merge(
        tracks_df,
        artists_df,
        left_on="id_artists",  # adjust if column name differs
        right_on="id",
        how="left",
        suffixes=("_track", "_artist")
    )

    # Drop redundant columns if needed
    if 'id_artist' in merged_df.columns:
        merged_df.drop(columns=['id_artist'], inplace=True)

    # Save merged dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path, index=False)
    print(f"✅ Merged dataset saved to {save_path}")
    return merged_df

if __name__ == "__main__":
    df_merged = merge_tracks_artists()
    print("✅ Merge completed successfully.")
