import numpy as np
import matplotlib.pyplot as plt


def generate_rf_stream(samples: int = 1000, inject_anomaly: bool = False):
    """Generates synthetic raw data representing a radio-frequency-like stream."""
    # 1. Create the timeline.
    time = np.linspace(0, 1, samples)

    # 2. Generate baseline normal communication (Gaussian noise + a 50 Hz carrier wave).
    background_noise = np.random.normal(0, 0.5, samples)
    carrier_wave = np.sin(2 * np.pi * 50 * time)
    rf_signal = carrier_wave + background_noise

    # 3. Inject a high-energy threat burst.
    if inject_anomaly:
        # Threat happens between 40% and 45% of the timeframe.
        burst_start = int(samples * 0.4)
        burst_end = int(samples * 0.45)

        # High-frequency, high-amplitude burst.
        threat_burst = 4 * np.sin(2 * np.pi * 300 * time[burst_start:burst_end])
        rf_signal[burst_start:burst_end] += threat_burst

    return time, rf_signal


def visualize_stream() -> None:
    """Visualizes the normal stream vs the threatened stream."""
    time, normal_signal = generate_rf_stream(inject_anomaly=False)
    _, threat_signal = generate_rf_stream(inject_anomaly=True)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Normal RF Communication (Baseline)")
    plt.plot(time, normal_signal, color="blue", alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.title("Anomalous Burst Detected (Threat Injection)")
    plt.plot(time, threat_signal, color="red", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_stream()
