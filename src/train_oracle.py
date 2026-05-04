import os
# Ensure we use JAX backend
import oracle_setup

import keras
import jax
import numpy as np
import matplotlib.pyplot as plt

from architecture import prepare_sequence_data, PhysicsAutoencoder

def main():
    print(f"Using Keras backend: {keras.backend.backend()}")
    print(f"Available JAX devices: {jax.devices()}")

    # 1. Load the data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'corrupted_signal.npy')
    if not os.path.exists(data_path):
        print(f"Error: Could not find data at {data_path}. Please run data_generator.py first.")
        return

    print("Loading data...")
    signal = np.load(data_path)
    
    # 2. Prepare the data
    window_size = 50
    print(f"Preparing sequence data with window size {window_size}...")
    train_data, test_data = prepare_sequence_data(signal, window_size=window_size)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # 3. Instantiate and compile the model
    print("Initializing PhysicsAutoencoder...")
    # Use JAX via Keras 3 backend
    model = PhysicsAutoencoder(latent_dim=8, original_dim=window_size)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    
    # 4. Train the model
    print("Training the Oracle on normal physics (before period 60)...")
    # In an autoencoder, inputs and targets are the same
    history = model.fit(
        train_data, train_data,
        epochs=15,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )
    
    # Save the model
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'physics_autoencoder.weights.h5')
    model.save_weights(model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # 5. Anomaly Detection
    print("Detecting anomalies across the entire signal...")
    # Re-window the entire signal to get predictions across time
    all_windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
    
    print("Generating reconstructions...")
    reconstructions = model.predict(all_windows, batch_size=512)
    
    # Calculate Mean Squared Error (MSE) for each window
    mse = np.mean(np.square(all_windows - reconstructions), axis=1)
    
    # 6. Plotting the results
    print("Plotting reconstruction errors...")
    plt.figure(figsize=(15, 6))
    
    # Create time axis for the windows (accounting for window size offset)
    T = 11.0 # period used in generator
    points_per_period = 1000
    total_time = T * 100 # 100 periods
    total_points = points_per_period * 100
    t = np.linspace(0, total_time, total_points)
    t_windows = t[window_size - 1:] # Time corresponding to the end of each window
    
    plt.plot(t_windows, mse, color='red', linewidth=1.5, label='Reconstruction Error (MSE)')
    plt.axvline(x=60 * T, color='green', linestyle='--', label='Training Split (Period 60)')
    plt.axvline(x=70 * T, color='black', linestyle=':', label='Sabotage Start (Period 70)')
    plt.axvline(x=75 * T, color='black', linestyle=':', label='Sabotage End (Period 75)')
    
    plt.title("Genesis Oracle: Anomaly Detection via Reconstruction Error")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.yscale("log") # Log scale helps visualize the massive spike
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'anomaly_detection.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Anomaly detection plot saved to {plot_path}")

if __name__ == "__main__":
    main()
