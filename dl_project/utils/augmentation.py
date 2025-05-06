import numpy as np

def add_gaussian_noise(signal, noise_factor=0.05):
    """Add random Gaussian noise to the signal"""
    noise = np.random.normal(0, noise_factor, signal.shape)
    return signal + noise

def time_shift(signal, shift_range=0.2):
    """Randomly shift the signal in time"""
    shift = int(signal.shape[0] * np.random.uniform(-shift_range, shift_range))
    return np.roll(signal, shift, axis=0)

def scale_amplitude(signal, scale_factor_range=(0.8, 1.2)):
    """Randomly scale the signal amplitude"""
    scale = np.random.uniform(*scale_factor_range)
    return signal * scale

def augment_1d(signal_batch, augment_prob=0.5):
    """Apply random augmentations to a batch of 1D signals"""
    augmented = np.copy(signal_batch)
    
    for i in range(len(augmented)):
        if np.random.random() < augment_prob:
            # Apply random combinations of augmentations
            if np.random.random() < 0.5:
                augmented[i] = add_gaussian_noise(augmented[i])
            if np.random.random() < 0.5:
                augmented[i] = time_shift(augmented[i])
            if np.random.random() < 0.5:
                augmented[i] = scale_amplitude(augmented[i])
    
    return augmented

def plot_augmentation_examples(original_signal, n_examples=3):
    """Plot original signal vs augmented versions"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 3*n_examples))
    plt.subplot(n_examples+1, 1, 1)
    plt.plot(original_signal)
    plt.title('Original Signal')
    
    for i in range(n_examples):
        augmented = augment_1d(original_signal[np.newaxis, :])[0]
        plt.subplot(n_examples+1, 1, i+2)
        plt.plot(augmented)
        plt.title(f'Augmented Example {i+1}')
    
    plt.tight_layout()
    plt.show()
