import numpy as np
import matplotlib.pyplot as plt
import os

def generate_square_wave_fourier(T=11.0, num_periods=100, num_harmonics=9, points_per_period=1000):
    """
    Generates the Fourier series of a square wave using the first `num_harmonics` odd harmonics.
    """
    omega_0 = 2 * np.pi / T
    total_time = T * num_periods
    total_points = points_per_period * num_periods
    
    t = np.linspace(0, total_time, total_points)
    y = np.zeros_like(t)
    
    for n in range(num_harmonics):
        k = 2 * n + 1  # Odd harmonics: 1, 3, 5, 7, 9, 11, 13, 15, 17
        amplitude = 4 / (k * np.pi)
        y += amplitude * np.sin(k * omega_0 * t)
        
    return t, y

def apply_rc_filter(t, T=11.0, num_harmonics=9, R=500.0, C=1000e-6):
    """
    Applies the analytical RC filter to the Fourier series components.
    Calculates amplitude attenuation and phase shift for each harmonic.
    """
    omega_0 = 2 * np.pi / T
    y_filtered = np.zeros_like(t)
    
    for n in range(num_harmonics):
        k = 2 * n + 1
        omega_k = k * omega_0
        
        # Original amplitude
        A_k = 4 / (k * np.pi)
        
        # Filter transfer function H(w) = 1 / (1 + j * w * R * C)
        H_denom = 1 + 1j * omega_k * R * C
        H_k = 1 / H_denom
        
        # Filter effect
        amp_attenuation = np.abs(H_k)
        phase_shift = np.angle(H_k)
        
        # Apply to harmonic
        A_filtered = A_k * amp_attenuation
        y_filtered += A_filtered * np.sin(omega_k * t + phase_shift)
        
    return y_filtered

def inject_noise_and_sabotage(t, y_filtered, T=11.0, noise_std=0.05):
    """
    Adds Gaussian noise and injects a massive high-frequency voltage spike 
    between period 70 and 75.
    """
    # 1. Add random Gaussian noise
    noise = np.random.normal(0, noise_std, size=len(y_filtered))
    y_corrupted = y_filtered + noise
    
    # 2. Inject massive high-frequency voltage spike between period 70 and 75
    start_time = 70 * T
    end_time = 75 * T
    
    spike_mask = (t >= start_time) & (t <= end_time)
    
    # Generate high-frequency spike (e.g., 20x fundamental freq, amplitude 5.0)
    high_freq_omega = 20 * (2 * np.pi / T)
    spike_amplitude = 5.0
    spike_signal = spike_amplitude * np.sin(high_freq_omega * t[spike_mask])
    
    # Add the spike to the masked region
    y_corrupted[spike_mask] += spike_signal
    
    return y_corrupted

def main():
    # Parameters
    T = 11.0
    num_periods = 100
    num_harmonics = 9
    points_per_period = 1000
    
    # --- CHANGE THIS TO YOUR ACTUAL LAST 3 ID DIGITS ---
    last_3_id_digits = 000  
    
    R = 0.5 * 1000  # 0.5 kOhm
    C = (1000 + last_3_id_digits) * 1e-6  # in Farads
    
    print(f"Generating continuous Fourier series of a square wave...")
    t, y_original = generate_square_wave_fourier(T=T, num_periods=num_periods, 
                                                 num_harmonics=num_harmonics, 
                                                 points_per_period=points_per_period)
    
    print(f"Applying RC low-pass filter (R={R} ohms, C={C} F)...")
    y_filtered = apply_rc_filter(t, T=T, num_harmonics=num_harmonics, R=R, C=C)
    
    print(f"Injecting noise and sabotage (spike between period 70 and 75)...")
    y_corrupted = inject_noise_and_sabotage(t, y_filtered, T=T, noise_std=0.05)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save the 1D continuous array (corrupted signal)
    data_path = "data/corrupted_signal.npy"
    np.save(data_path, y_corrupted)
    print(f"Saved corrupted 1D signal to {data_path}")
    
    # Save a preview plot to verify
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Original vs Filtered
    plt.subplot(3, 1, 1)
    points_to_plot = 3 * points_per_period
    plt.plot(t[:points_to_plot], y_original[:points_to_plot], label="Original Square Wave", alpha=0.7)
    plt.plot(t[:points_to_plot], y_filtered[:points_to_plot], label="Filtered Signal", linewidth=2)
    plt.title("Original vs Filtered Signal (First 3 Periods)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Zoom on the Spike
    plt.subplot(3, 1, 2)
    start_idx = 68 * points_per_period
    end_idx = 77 * points_per_period
    plt.plot(t[start_idx:end_idx], y_corrupted[start_idx:end_idx], color='red', linewidth=1)
    plt.title("Corrupted Signal with Spike (Periods 68 to 77)")
    plt.axvline(70*T, color='k', linestyle='--', label="Spike Start (t=770)")
    plt.axvline(75*T, color='k', linestyle='--', label="Spike End (t=825)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Full Signal
    plt.subplot(3, 1, 3)
    plt.plot(t, y_corrupted, color='purple', linewidth=0.5)
    plt.title("Full Corrupted Signal (100 Periods)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = "data/filter_and_noise_preview.png"
    plt.savefig(plot_path)
    print(f"Saved preview plot to {plot_path}")

if __name__ == "__main__":
    main()
