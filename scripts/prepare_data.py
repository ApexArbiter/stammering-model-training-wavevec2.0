import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_dataset_csv():
    """
    Create CSV file with audio paths and labels
    """
    data = []

    # Check if the directories exist and contain files
    stammer_dir = Path("data/stammering")
    normal_dir = Path("data/non_stammering")

    if not stammer_dir.exists():
        raise FileNotFoundError(f"Directory {stammer_dir} does not exist.")
    if not normal_dir.exists():
        raise FileNotFoundError(f"Directory {normal_dir} does not exist.")

    print(f"Found {len(list(stammer_dir.glob('*.wav')))} stammering files.")
    print(f"Found {len(list(normal_dir.glob('*.wav')))} non-stammering .wav files.")
    print(f"Found {len(list(normal_dir.glob('*.mp3')))} non-stammering .mp3 files.")

    # Process stammering files
    for audio_file in stammer_dir.glob("*.wav"):
        print(f"Processing stammering file: {audio_file}")  # Debug print
        data.append({
            'file_path': str(audio_file),
            'label': 1,  # 1 for stammering
            'label_name': 'stammering'
        })

    # Process non-stammering files (both .mp3 and .wav)
    for audio_file in normal_dir.glob("*.wav"):
        print(f"Processing non-stammering .wav file: {audio_file}")  # Debug print
        data.append({
            'file_path': str(audio_file),
            'label': 0,  # 0 for non-stammering
            'label_name': 'non_stammering'
        })
    for audio_file in normal_dir.glob("*.mp3"):
        print(f"Processing non-stammering .mp3 file: {audio_file}")  # Debug print
        data.append({
            'file_path': str(audio_file),
            'label': 0,  # 0 for non-stammering
            'label_name': 'non_stammering'
        })

    # Check if no data was found
    if not data:
        raise ValueError("No audio files found in 'stammering' or 'non_stammering' directories.")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Check if the 'label' column exists
    if 'label' not in df.columns:
        raise KeyError("The 'label' column is missing!")

    print(f"DataFrame created with {len(df)} samples.")
    print(f"First few rows:\n{df.head()}")

    # Split into train/validation/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine and save
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv('data/dataset.csv', index=False)

    print(f"Dataset created:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Stammering: {len(df[df['label'] == 1])} samples")
    print(f"Non-stammering: {len(df[df['label'] == 0])} samples")

    return final_df

if __name__ == "__main__":
    # Create directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create dataset CSV
    df = create_dataset_csv()
