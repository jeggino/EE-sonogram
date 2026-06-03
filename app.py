import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import stft
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Interactive Bat Sonogram – Pro Interface")

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload ultrasonic audio", type=["wav", "flac", "mp3"])

if not uploaded_file:
    st.info("Upload an ultrasonic recording to begin (supports long files, explored in 5 s chunks).")
    st.stop()

raw = uploaded_file.read()
data, sr = sf.read(io.BytesIO(raw))
if data.ndim > 1:
    data = data.mean(axis=1)
y = data.astype(float)
duration = len(y) / sr

st.write(f"Sample rate: {sr} Hz, duration: {duration:.3f} s")

# =========================================================
# CHUNKING FOR LONG RECORDINGS
# =========================================================
chunk_size = 5.0  # seconds per chunk
num_chunks = int(np.ceil(duration / chunk_size))

chunk_labels = [f"{i+1} chunk" if i == 0 else f"{i+1} chunk" for i in range(num_chunks)]
# More readable labels:
chunk_labels = [f"{i+1} chunk" for i in range(num_chunks)]
chunk_labels[0] = "First chunk"
if num_chunks > 1:
    chunk_labels[1] = "Second chunk"
if num_chunks > 2:
    chunk_labels[2] = "Third chunk"

# =========================================================
# LAYOUT
# =========================================================
col_controls, col_plot = st.columns([1, 2])

# =========================================================
# LEFT COLUMN – CONTROLS
# =========================================================
with col_controls:

    # Spectrogram settings
    with st.expander("⚙️ Spectrogram settings", expanded=True):
        # Fixed FFT parameters
        n_fft = 1024
        hop = 256

        # Chunk selection as selectbox
        selected_chunk_label = st.selectbox(
            "Select chunk (5 s each)",
            chunk_labels,
            index=0,
        )
        chunk_index = chunk_labels.index(selected_chunk_label)
        chunk_start = chunk_index * chunk_size
        chunk_end = min(duration, chunk_start + chunk_size)

        st.write(f"Showing {selected_chunk_label}: {chunk_start:.2f}–{chunk_end:.2f} s")

        # Window start within chunk (global time)
        start_min = float(chunk_start)
        start_max = float(chunk_end)
        start_time = st.slider(
            "Window start (s, global time)",
            start_min,
            start_max,
            start_min,
        )

        # Frequency zoom
        min_khz = st.slider("Min frequency (kHz)", 5, 80, 15)
        max_khz = st.slider("Max frequency (kHz)", int(min_khz), int(sr / 2000), 120)

        # Display mode moved here
        mode = st.radio("Display mode", ["Scatter", "Heatmap"], index=0)

    # Amplitude filtering
    with st.expander("🔊 Amplitude filtering", expanded=True):
        amp_cut = st.slider("Minimum amplitude (dB)", -120, 0, -80)
        keep_top_percent = st.slider("Keep top (%) strongest points", 1, 50, 10)

# =========================================================
# RIGHT COLUMN – SONOGRAM + INFO
# =========================================================
with col_plot:
    # Extract current chunk
    start_sample = int(chunk_start * sr)
    end_sample = int(chunk_end * sr)
    if end_sample <= start_sample:
        st.error("Selected chunk has no data.")
        st.stop()
    y_chunk = y[start_sample:end_sample]

    # ---------- Compute STFT on current chunk ----------
    f, t_local, Zxx = stft(y_chunk, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    t = t_local + chunk_start  # global time
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)

    # Time window = whole chunk (start_time is just a reference marker)
    time_mask = (t >= chunk_start) & (t <= chunk_end)
    t_sel = t[time_mask]
    S_db_sel = S_db[:, time_mask]

    # Frequency zoom mask
    f_min = min_khz * 1000
    f_max = max_khz * 1000
    freq_mask = (f >= f_min) & (f <= f_max)
    f_sel = f[freq_mask]
    S_db_sel = S_db_sel[freq_mask, :]

    if S_db_sel.size == 0:
        st.error("No data in selected frequency band.")
        st.stop()

    # ---------- Flatten for scatter ----------
    T, F = np.meshgrid(t_sel, f_sel)
    time_vals = T.flatten()
    freq_vals = F.flatten()
    amp_vals = S_db_sel.flatten()

    # Amplitude filter
    amp_mask = amp_vals >= amp_cut
    time_vals = time_vals[amp_mask]
    freq_vals = freq_vals[amp_mask]
    amp_vals = amp_vals[amp_mask]

    if len(amp_vals) == 0:
        st.warning("No points above amplitude threshold in this window/band.")
        st.stop()

    # Keep top X%
    perc = 100 - keep_top_percent
    threshold = np.percentile(amp_vals, perc)
    strong_mask = amp_vals >= threshold
    time_vals = time_vals[strong_mask]
    freq_vals = freq_vals[strong_mask]
    amp_vals = amp_vals[strong_mask]

    # Downsample if needed
    max_points = 200_000
    if len(time_vals) > max_points:
        idx = np.random.choice(len(time_vals), max_points, replace=False)
        time_vals = time_vals[idx]
        freq_vals = freq_vals[idx]
        amp_vals = amp_vals[idx]

    # ---------- Base figure ----------
    if mode == "Scatter":
        fig = px.scatter(
            x=time_vals,
            y=freq_vals,
            color=amp_vals,
            color_continuous_scale="magma",
            render_mode="webgl",
            opacity=0.6,
            labels={"x": "Time (s)", "y": "Frequency (Hz)", "color": "Amplitude (dB)"},
            title="Interactive Call-Focused Sonogram",
        )
        fig.update_traces(marker=dict(size=2))
    else:
        fig = go.Figure(
            data=go.Heatmap(
                x=t_sel,
                y=f_sel,
                z=S_db_sel,
                colorscale="magma",
                colorbar=dict(title="Amplitude (dB)"),
            )
        )
        fig.update_layout(
            title="Heatmap Sonogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
        )

    fig.update_layout(
        height=700,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[f_min, f_max]),
    )

    # ---------- Plot ----------
    fig_container = st.container()
    fig_container.plotly_chart(fig, use_container_width=True)

    # ---------- Distance between two points in time ----------
    st.markdown("### ⏱ Time distance between two points")
    col_a, col_b = st.columns(2)
    with col_a:
        t_a = st.number_input(
            "Point A time (s)",
            min_value=float(chunk_start),
            max_value=float(chunk_end),
            value=float(chunk_start),
            step=0.001,
            format="%.3f",
        )
    with col_b:
        t_b = st.number_input(
            "Point B time (s)",
            min_value=float(chunk_start),
            max_value=float(chunk_end),
            value=float(chunk_start),
            step=0.001,
            format="%.3f",
        )

    dt = abs(t_b - t_a)
    st.write(f"Time distance: **{dt:.4f} s**")

    st.markdown("### 🔍 Hover info")
    st.write("Hover over points to see time, frequency, and amplitude.")











