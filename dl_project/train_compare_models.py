import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from eeg_cnn_models import create_lenet, create_alexnet, create_resnet, create_googlenet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.augmentation import augment_1d, plot_augmentation_examples

class AugmentationSequence(tf.keras.utils.Sequence):
    """Custom sequence for applying augmentations during training"""
    def __init__(self, x, y, batch_size=64):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.indexes = np.arange(len(x))
        
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_idx]
        batch_y = self.y[batch_idx]
        
        # Apply augmentation only during training
        return augment_1d(batch_x), batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def load_and_preprocess_data(file_path, target_column, test_size=0.2):
    """Load and preprocess the EEG data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Prepare features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    # Convert to categorical
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes=num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Reshape for CNN input (add channel dimension)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    return X_train, X_test, y_train, y_test, num_classes

def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=100):
    """Train a model with augmentation and improved callbacks"""
    if not os.path.exists('model_checkpoints'):
        os.makedirs('model_checkpoints')
    
    # Create data sequences
    train_seq = AugmentationSequence(X_train, y_train, batch_size=64)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            f'model_checkpoints/{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Plot augmentation examples
    plot_augmentation_examples(X_train[0, :, 0])
    
    # Train with augmentation sequence
    history = model.fit(
        train_seq,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks
    )
    
    return history

def plot_training_history(histories, model_names):
    """Plot training histories for multiple models"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['accuracy'], label=f'{name} (Train)')
        plt.plot(history.history['val_accuracy'], linestyle='--', 
                label=f'{name} (Val)')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['loss'], label=f'{name} (Train)')
        plt.plot(history.history['val_loss'], linestyle='--',
                label=f'{name} (Val)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(models, X_test, y_test, model_names):
    """Plot confusion matrices for all models"""
    n_models = len(models)
    fig, axes = plt.subplots(2, (n_models+1)//2, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set parameters
    EPOCHS = 100
    
    # Load and preprocess data
    # For 7-class classification
    X_train_7, X_test_7, y_train_7, y_test_7, num_classes_7 = load_and_preprocess_data(
        'EEG_final_preprocessed_99PCA.csv', 'main.disorder'
    )
    
    # Create models
    input_shape = (X_train_7.shape[1], 1)
    models_7 = [
        create_lenet(input_shape, num_classes_7),
        create_alexnet(input_shape, num_classes_7),
        create_resnet(input_shape, num_classes_7),
        create_googlenet(input_shape, num_classes_7)
    ]
    model_names = ['LeNet', 'AlexNet', 'ResNet', 'GoogLeNet']
    
    # Train models and collect histories
    histories_7 = []
    for model, name in zip(models_7, model_names):
        print(f"\nTraining {name} for 7-class classification...")
        history = train_model(model, X_train_7, y_train_7, X_test_7, y_test_7, 
                            f"{name}_7class", EPOCHS)
        histories_7.append(history)
    
    # Plot results
    plot_training_history(histories_7, model_names)
    plot_confusion_matrices(models_7, X_test_7, y_test_7, model_names)
    
    # Print classification reports
    for model, name in zip(models_7, model_names):
        print(f"\nClassification Report for {name} (7-class):")
        y_pred = model.predict(X_test_7)
        print(classification_report(y_test_7.argmax(axis=1), y_pred.argmax(axis=1)))

if __name__ == "__main__":
    main()
