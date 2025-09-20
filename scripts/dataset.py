import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import librosa
import numpy as np

class StammeringDataset(Dataset):
    def __init__(self, csv_file, split='train', max_length=160000):  # 10 seconds at 16kHz
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        label = row['label']
        
        # Load and preprocess audio
        try:
            # Load audio with librosa for better compatibility
            audio, _ = librosa.load(file_path, sr=16000)
            
            # Convert to tensor
            audio = torch.FloatTensor(audio)
            
            # Pad or truncate
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.max_length - len(audio)))
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence if file can't be loaded
            audio = torch.zeros(self.max_length)
        
        return {
            'input_values': audio,
            'labels': torch.tensor(label, dtype=torch.long)
        }