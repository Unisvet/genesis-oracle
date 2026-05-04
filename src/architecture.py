import os
import numpy as np
import keras

def prepare_sequence_data(signal, window_size=50, split_period=60):
    """
    Slices a 1D signal into overlapping 2D matrices of shape (num_windows, window_size).
    Splits the data into training (before period 60) and testing sets.
    """
    signal = np.array(signal)
    
    # Create overlapping windows using sliding_window_view
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
    
    # Split data: normal data before period 60 for training, rest for testing
    train_data = windows[:split_period]
    test_data = windows[split_period:]
    
    return train_data, test_data

class SignalCompression(keras.layers.Layer):
    """
    Custom bottleneck layer that reduces a 50-timestep window to 8 latent dimensions.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        # Define a dense weight matrix and bias inside the build method
        self.w = self.add_weight(
            shape=(input_shape[-1], self.latent_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="compression_weights"
        )
        self.b = self.add_weight(
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
            name="compression_biases"
        )

    def call(self, inputs):
        # Forward pass with ReLU activation
        z = keras.ops.matmul(inputs, self.w) + self.b
        return keras.activations.relu(z)

class SignalExpansion(keras.layers.Layer):
    """
    Custom layer to expand the 8 latent dimensions back to the original 50-timestep window.
    """
    def __init__(self, original_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = original_dim

    def build(self, input_shape):
        # Define a dense weight matrix and bias for reconstruction
        self.w = self.add_weight(
            shape=(input_shape[-1], self.original_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="expansion_weights"
        )
        self.b = self.add_weight(
            shape=(self.original_dim,),
            initializer="zeros",
            trainable=True,
            name="expansion_biases"
        )

    def call(self, inputs):
        # Forward pass to reconstruct the original signal dimensions
        return keras.ops.matmul(inputs, self.w) + self.b

class PhysicsAutoencoder(keras.Model):
    """
    Autoencoder chaining the SignalCompression encoder to a SignalExpansion decoder.
    """
    def __init__(self, latent_dim=8, original_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(original_dim=original_dim)

    def call(self, inputs):
        # Encode and decode
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
