import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import io

st.title("Bat Sonogram Analyzer (No Supabase)")

uploaded_file = st.file_uploader("Upload a bat audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    # Load audio safely for Streamlit Cloud
    data, sr = sf.read(io.BytesIO(uploaded_file.read()))
    y = data.astype(float)

    st.write(f"Sample rate: {sr} Hz")
    st.audio(uploaded_file)

    # Spectrogram settings
    st.subheader("Spectrogram settings")
    n_fft = st.slider("FFT window size", 256, 4096, 1024)
    hop_length = st.slider("Hop length", 64, 1024, 256)
    max_freq_khz = st.slider("Max frequency (kHz)", 20, int(sr / 2000), 120)

    # Compute spectrogram
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 5))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax,
    )
    ax.set_ylim(0, max_freq_khz * 1000)
    ax.set_title("Spectrogram")
    st.pyplot(fig)
    plt.close(fig)

    # Automatic call detection
    st.subheader("Automatic call detection")

    min_khz = st.slider("Min frequency (kHz)", 5, int(max_freq_khz), 15)
    max_khz = st.slider("Max frequency (kHz)", int(min_khz), int(max_freq_khz), 120)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= min_khz * 1000) & (freqs <= max_khz * 1000)

    band_energy = S_mag[band_mask, :].mean(axis=0)

    threshold_factor = st.slider("Threshold factor", 1.0, 10.0, 3.0)
    threshold = threshold_factor * np.median(band_energy)

    call_frames = band_energy > threshold

    calls = []
    in_call = False
    start_idx = None

    for i, is_call in enumerate(call_frames):
        if is_call and not in_call:
            in_call = True
            start_idx = i
        elif not is_call and in_call:
            in_call = False
            end_idx = i
            calls.append((start_idx, end_idx))

    if in_call and start_idx is not None:
        calls.append((start_idx, len(call_frames)))

    times = librosa.frames_to_time(np.arange(len(call_frames)), sr=sr, hop_length=hop_length)

    st.write(f"Detected calls: {len(calls)}")

    for idx, (s_idx, e_idx) in enumerate(calls, start=1):
        start_time = times[s_idx]
        end_time = times[e_idx - 1] if e_idx - 1 < len(times) else times[-1]
        duration = end_time - start_time
        st.write(f"Call {idx}: {start_time:.3f}s – {end_time:.3f}s (duration {duration:.3f}s)")

