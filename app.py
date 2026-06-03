import os
import io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
from supabase import create_client, Client
from datetime import datetime

# -----------------------------
# Supabase setup
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("Bat Sonogram Analyzer")

st.markdown(
    "Upload ultrasonic bat recordings (e.g. 384 kHz) to visualize spectrograms, "
    "detect calls, and store results in Supabase."
)

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "flac", "mp3"])

if uploaded_file:
    # -----------------------------
    # Load audio (keep original sample rate)
    # -----------------------------
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    st.write(f"Sample rate: {sr} Hz")
    st.audio(uploaded_file)

    # -----------------------------
    # Spectrogram parameters
    # -----------------------------
    st.subheader("Spectrogram settings")
    n_fft = st.slider("FFT window size", 256, 4096, 1024, step=256)
    hop_length = st.slider("Hop length", 64, 2048, 256, step=64)
    max_freq_khz = st.slider("Max frequency to display (kHz)", 20, int(sr / 2000), 120)

    # -----------------------------
    # Compute spectrogram
    # -----------------------------
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
    ax.set_title("Spectrogram (Sonogram)")
    st.pyplot(fig)

    # -----------------------------
    # Simple automatic call detection
    # -----------------------------
    st.subheader("Automatic call detection")

    # Frequency band for bats (e.g. 15–120 kHz)
    min_khz = st.slider("Min frequency (kHz)", 5, int(max_freq_khz), 15)
    max_khz = st.slider("Max frequency (kHz)", int(min_khz), int(max_freq_khz), 120)

    # Convert to bin indices
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= min_khz * 1000) & (freqs <= max_khz * 1000)

    band_energy = S_mag[band_mask, :].mean(axis=0)

    # Threshold as a multiple of median energy
    threshold_factor = st.slider("Threshold factor (x median)", 1.0, 10.0, 3.0, 0.5)
    threshold = threshold_factor * np.median(band_energy)

    call_frames = band_energy > threshold

    # Group consecutive frames into calls
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

    # -----------------------------
    # Store results in Supabase
    # -----------------------------
    if supabase is None:
        st.info("Supabase credentials not set. Skipping database storage.")
    else:
        st.subheader("Store results in Supabase")

        project_name = st.text_input("Project / recording name", value=uploaded_file.name)
        table_name = "bat_calls"  # adjust to your schema

        if st.button("Save detections to Supabase"):
            rows = []
            for idx, (s_idx, e_idx) in enumerate(calls, start=1):
                start_time = times[s_idx]
                end_time = times[e_idx - 1] if e_idx - 1 < len(times) else times[-1]
                duration = end_time - start_time
                rows.append(
                    {
                        "project": project_name,
                        "call_index": idx,
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "duration": float(duration),
                        "sample_rate": int(sr),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )

            if rows:
                try:
                    res = supabase.table(table_name).insert(rows).execute()
                    st.success(f"Saved {len(rows)} calls to Supabase table '{table_name}'.")
                except Exception as e:
                    st.error(f"Error saving to Supabase: {e}")
            else:
                st.warning("No calls detected to save.")

    ax.set_title("Spectrogram (Sonogram)")
    st.pyplot(fig)
