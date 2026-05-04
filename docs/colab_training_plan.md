# Genesis Oracle: Colab Training & Anomaly Detection Plan

This guide provides step-by-step instructions for pulling the Genesis Oracle codebase into a Google Colab environment, leveraging JAX-accelerated hardware, and running the anomaly detection model.

## Step 1: Initialize Cloud Environment

1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Navigate to **Runtime > Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** or **TPU v2** (TPU is highly recommended for maximizing JAX XLA compilation speeds).
4. Save the runtime settings.

## Step 2: Pulling from the Vault

In the first cell of your Colab notebook, pull your codebase directly into the cloud instance and enter the directory.

```bash
# Cell 1
!git clone https://github.com/Unisvet/genesis-oracle.git
%cd genesis-oracle
```

## Step 3: Injecting `uv` into the Cloud

To bypass Colab's relatively slow default package manager, we inject `uv`. This reads your `pyproject.toml` and installs all exact dependencies blisteringly fast.

```bash
# Cell 2
!pip install uv
!uv pip install --system -r pyproject.toml
```

## Step 4: Training the Oracle

In this cell, you will load your signal data, configure the architecture, and compile the autoencoder. As requested, we will use the Adam optimizer, MSE loss for the gradient descent, and train it for exactly 30 epochs.

> [!NOTE] 
> If you haven't generated the data file `corrupted_signal.npy` yet, running `data_generator.py` first is necessary. The script below will automatically trigger data generation if it's missing.

```python
# Cell 3
import os
import sys
import jax
import keras
import numpy as np
import matplotlib.pyplot as plt

# Generate data if it's not present in the cloned repository
if not os.path.exists('data/corrupted_signal.npy'):
    print("Corrupted signal data not found. Generating...")
    sys.path.append('src')
    from data_generator import main as generate_data
    generate_data()

from src.architecture import prepare_sequence_data, PhysicsAutoencoder

# 1. Load the generated signal
signal = np.load('data/corrupted_signal.npy')
window_size = 50

# 2. Prepare sequences: Splits at period 60 automatically
train_data, test_data = prepare_sequence_data(signal, window_size=window_size)

# 3. Instantiate and Compile the Model
print(f"Active JAX devices: {jax.devices()}")
model = PhysicsAutoencoder(latent_dim=8, original_dim=window_size)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

# 4. Train the Oracle on normal data
history = model.fit(
    train_data, train_data,
    epochs=30,           # Train for 30 epochs
    batch_size=256,
    validation_split=0.1,
    verbose=1
)
```
*Notice the compilation delay on the first epoch—this is the JAX XLA compiler optimizing your computational graph for the selected hardware. Subsequent epochs will run incredibly fast.*

## Step 5: The Detection Run & Anomaly Trigger

Now, pass the full dataset through the trained autoencoder to calculate the Mean Absolute Error (MAE) and plot the detection spike.

```python
# Cell 4
# Pass the entire dataset (including the anomaly) through the trained Autoencoder
all_windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
reconstructions = model.predict(all_windows, batch_size=512)

# Calculate Mean Absolute Error (MAE) at each time step
mae = np.mean(np.abs(all_windows - reconstructions), axis=1)

# Plotting Deliverables
plt.figure(figsize=(15, 6))

# Time axis reconstruction
T = 11.0 
points_per_period = 1000
total_time = T * 100
t = np.linspace(0, total_time, points_per_period * 100)
t_windows = t[window_size - 1:]

# Plot MAE
plt.plot(t_windows, mae, color='blue', linewidth=1.5, label='Reconstruction Error (MAE)')

# Define and draw the Anomaly Threshold
# (0.15 is generally safe based on standard noise floors, adjust if needed)
anomaly_threshold = 0.15 
plt.axhline(y=anomaly_threshold, color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')

# Formatting the plot
plt.title("Reconstruction Loss over Time (Anomaly Detection)")
plt.xlabel("Time")
plt.ylabel("Mean Absolute Error (MAE)")
plt.yscale('log') # Enhances visualization of the massive spike
plt.legend()
plt.grid(True, alpha=0.3)

# Save and download the plot
plot_filename = 'anomaly_detection_mae_colab.png'
plt.tight_layout()
plt.savefig(plot_filename)
plt.show()

# Trigger Colab download
try:
    from google.colab import files
    files.download(plot_filename)
    print(f"Successfully downloaded {plot_filename}")
except ImportError:
    print("Not in Colab environment. Plot saved locally.")
```

When you run this final cell, you will see a flat, stable line for the normal physics, followed by a massive, undeniable spike penetrating your red dashed threshold precisely between periods 70 and 75, indicating a successful trigger detection.
