import torch
import librosa
import numpy as np
from train_model import StammeringClassifier

class StammeringPredictor:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StammeringClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, audio_path):
        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # Pad or truncate to 10 seconds
        max_length = 160000  # 10 seconds at 16kHz
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(audio_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        result = {
            'prediction': 'stammering' if prediction == 1 else 'non_stammering',
            'confidence': confidence,
            'probabilities': {
                'non_stammering': probabilities[0][0].item(),
                'stammering': probabilities[0][1].item()
            }
        }
        
        return result

# Example usage
if __name__ == "__main__":
    predictor = StammeringPredictor()
    
    # Test on a single file
    audio_file = "path/to/your/test_audio.wav"
    result = predictor.predict(audio_file)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")