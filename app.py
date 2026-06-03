import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import stft
import plotly.express as px

st.title("Interactive Bat Sonogram (Call-focused, Optimized)")

uploaded_file = st.file_uploader("Upload ultrasonic audio", type=["wav", "flac", "mp3"])

if uploaded_file:
    # Load audio
    raw = uploaded_file.read()
    data, sr = sf.read(io.BytesIO(raw))
    if data.ndim > 1:
        data = data.mean(axis=1)
    y = data.astype(float)

    st.write(f"Sample rate: {sr} Hz")
    st.audio(io.BytesIO(raw), format="audio/wav")

    # ---------------- Spectrogram settings ----------------
    st.subheader("Spectrogram settings")

    # Call-focused frequency band
    min_khz = st.slider("Min frequency (kHz)", 5, 80, 15)
    max_khz = st.slider("Max frequency (kHz)", int(min_khz), int(sr / 2000), 120)

    # STFT resolution (smaller = fewer points)
    n_fft = st.slider("FFT window size", 256, 4096, 1024, step=256)
    hop = st.slider("Hop length", 64, n_fft - 64, 256, step=64)

    # ---------------- Compute STFT ----------------
    f, t, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)

    # ---------------- Time window limit ----------------
    total_time = float(t[-1]) if len(t) > 0 else 0.0
    if total_time == 0:
        st.error("Audio too short or STFT failed.")
        st.stop()

    max_time = st.slider("Max time window (s)", 0.1, max(0.2, total_time), min(5.0, total_time))
    time_mask = t <= max_time

    t_sel = t[time_mask]
    S_db_sel = S_db[:, time_mask]

    # ---------------- Call-focused frequency band ----------------
    f_min = min_khz * 1000
    f_max = max_khz * 1000
    freq_mask = (f >= f_min) & (f <= f_max)

    f_sel = f[freq_mask]
    S_db_sel = S_db_sel[freq_mask, :]

    # ---------------- Flatten to scatter points ----------------
    T, F = np.meshgrid(t_sel, f_sel)
    time_vals = T.flatten()
    freq_vals = F.flatten()
    amp_vals = S_db_sel.flatten()

    # ---------------- Keep only strongest points ----------------
    st.subheader("Density control")
    keep_top_percent = st.slider("Keep top (%) strongest points", 1, 50, 10)
    perc = 100 - keep_top_percent
    threshold = np.percentile(amp_vals, perc)

    mask = amp_vals >= threshold
    time_vals = time_vals[mask]
    freq_vals = freq_vals[mask]
    amp_vals = amp_vals[mask]

    st.write(f"Points plotted: {len(time_vals):,}")

    # Safety: if still huge, downsample further
    max_points = 200_000
    if len(time_vals) > max_points:
        idx = np.random.choice(len(time_vals), max_points, replace=False)
        time_vals = time_vals[idx]
        freq_vals = freq_vals[idx]
        amp_vals = amp_vals[idx]
        st.info(f"Downsampled to {max_points:,} points for performance.")

    # ---------------- Plotly scatter sonogram ----------------
    fig = px.scatter(
        x=time_vals,
        y=freq_vals,
        color=amp_vals,
        color_continuous_scale="magma",
        render_mode="webgl",
        opacity=0.7,
        labels={"x": "Time (s)", "y": "Frequency (Hz)", "color": "Amplitude (dB)"},
        title="Interactive Call-focused Sonogram",
    )

    fig.update_layout(
        height=700,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[f_min, f_max]),
    )

    st.plotly_chart(fig, use_container_width=True)




