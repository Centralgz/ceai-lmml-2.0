from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

AUDIO_FILE = "challenge.wav"
OUTPUT_IMAGE = "spectrogram.png"


def main():
    sample_rate, data = wavfile.read(AUDIO_FILE)

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    print(f"Sample rate : {sample_rate} Hz")
    print(f"Duration    : {len(data) / sample_rate:.2f} s")
    print(f"Samples     : {len(data)}")

    plt.figure(figsize=(14, 6))

    Pxx, freqs, bins, im = plt.specgram(
        data,
        NFFT=2048,
        Fs=sample_rate,
        noverlap=1024,
        cmap="viridis",
        scale="dB"
    )

    plt.title("Spectrogram – Hidden Message", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"\nSpectrogram saved to: {OUTPUT_IMAGE}")

    plt.show()


if __name__ == "__main__":
    main()