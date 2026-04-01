import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

# 🔹 Auto-detect device & sample rate
device = 14
sd.default.device = (device, None)

device_info = sd.query_devices(device, 'input')
fs = int(device_info['default_samplerate'])

def calculate_db(signal):
    rms = np.sqrt(np.mean(signal**2))
    
    if rms < 1e-8:
        return -100
    
    db = 20 * np.log10(rms)
    
    # Shift into human-friendly range
    db = db + 90
    
    return db

def get_frequency(signal):
    N = len(signal)

    # Windowing
    signal = signal * np.hanning(N)

    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, 1/fs)

    # Remove very low frequencies
    yf[xf < 20] = 0

    peak_index = np.argmax(yf)
    peak_freq = xf[peak_index]

    return peak_freq, xf, yf, peak_index

def reduce_noise(signal, noise):
    # Match size
    if len(noise) < len(signal):
        noise = np.tile(noise, int(len(signal)/len(noise)+1))
    noise = noise[:len(signal)]

    signal_fft = rfft(signal)
    noise_fft = rfft(noise)

    cleaned_fft = signal_fft - noise_fft
    cleaned_fft = np.maximum(cleaned_fft, 0)

    return irfft(cleaned_fft)

# 🔹 Input duration
duration = float(input("Enter duration (seconds): "))

# 🔹 Record noise
print("Recording noise... stay silent")
noise = sd.rec(int(fs * 2), samplerate=fs, channels=1)
sd.wait()
noise = noise.flatten()

# 🔹 Record sound
print("Recording sound... make noise")
audio = sd.rec(int(fs * duration), samplerate=fs, channels=1)
sd.wait()
audio = audio.flatten()

# 🔹 Normalize (VERY IMPORTANT FIX)
if np.max(np.abs(audio)) > 0:
    audio = audio / np.max(np.abs(audio))

# 🔹 Noise reduction
clean_audio = reduce_noise(audio, noise)

# 🔹 Ignore weak signals
if np.std(clean_audio) < 0.001:
    print("⚠️ Weak signal, try louder sound")

# 🔹 Analysis
freq, xf, yf, peak_idx = get_frequency(clean_audio)
db = calculate_db(clean_audio)

print(f"\n🎵 Frequency: {freq:.2f} Hz")
print(f"🔊 Volume: {db:.2f} dB")

# 🔹 Plot
plt.figure()
plt.plot(xf, yf)
plt.scatter(xf[peak_idx], yf[peak_idx])

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum")
plt.xlim(0, 2000)
plt.grid()

plt.show()
