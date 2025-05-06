import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

class EEGPredictor:
    def __init__(self, model_path):
        """Initialize the EEG predictor with a trained model"""
        self.model = load_model(model_path)
        self.class_labels = [
            'Control', 'Depression', 'ADHD', 'Schizophrenia', 
            'Alcoholic', 'Autism', 'OCD'
        ]
        
    def preprocess_signal(self, signal_data):
        """Preprocess the input signal data"""
        # Ensure the input is a numpy array
        signal = np.array(signal_data)
        
        # Reshape for model input (batch_size, timesteps, channels)
        signal = signal.reshape(1, -1, 1)
        
        return signal
    
    def predict(self, signal_data):
        """Predict the disorder class for the input signal"""
        # Preprocess the signal
        processed_signal = self.preprocess_signal(signal_data)
        
        # Make prediction
        prediction = self.model.predict(processed_signal)
        
        # Get the predicted class and probability
        predicted_class = np.argmax(prediction[0])
        probability = prediction[0][predicted_class]
        
        return {
            'disorder': self.class_labels[predicted_class],
            'probability': float(probability),
            'all_probabilities': {
                label: float(prob) 
                for label, prob in zip(self.class_labels, prediction[0])
            }
        }

def test_with_sample_data(predictor, data_path):
    """Test the model with sample data from a CSV file"""
    # Load sample data
    df = pd.read_csv(data_path)
    
    # Remove the target column if present
    if 'main.disorder' in df.columns:
        features = df.drop('main.disorder', axis=1)
    else:
        features = df
    
    # Test first sample
    sample = features.iloc[0].values
    result = predictor.predict(sample)
    
    print("Sample Prediction Results:")
    print(f"Predicted Disorder: {result['disorder']}")
    print(f"Confidence: {result['probability']:.2%}")
    print("\nProbabilities for all classes:")
    for disorder, prob in result['all_probabilities'].items():
        print(f"{disorder}: {prob:.2%}")

if __name__ == "__main__":
    # Initialize predictor with your best performing model
    predictor = EEGPredictor('model_checkpoints/LeNet_7class_best.keras')
    
    # Test with sample data
    test_with_sample_data(predictor, 'EEG_final_preprocessed_99PCA.csv')
    
    # Example of real-time prediction with custom input
    # Replace this with your actual signal data
    custom_signal = np.random.rand(99)  # Assuming 99 features after PCA
    result = predictor.predict(custom_signal)
    
    print("\nCustom Input Prediction:")
    print(f"Predicted Disorder: {result['disorder']}")
    print(f"Confidence: {result['probability']:.2%}")
