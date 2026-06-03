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
st.audio(io.BytesIO(raw), format="audio/wav")

# =========================================================
# CHUNKING FOR LONG RECORDINGS
# =========================================================
chunk_size = 5.0  # seconds per chunk
num_chunks = int(np.ceil(duration / chunk_size))

st.subheader("⏱️ Chunk navigation")
chunk_index = st.slider(
    "Select chunk (5 s each)",
    0,
    max(0, num_chunks - 1),
    0,
)
chunk_start = chunk_index * chunk_size
chunk_end = min(duration, chunk_start + chunk_size)

st.write(f"Showing chunk {chunk_index+1}/{num_chunks}: {chunk_start:.2f}–{chunk_end:.2f} s")

start_sample = int(chunk_start * sr)
end_sample = int(chunk_end * sr)
if end_sample <= start_sample:
    st.error("Selected chunk has no data.")
    st.stop()
y_chunk = y[start_sample:end_sample]
chunk_duration = chunk_end - chunk_start

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

        # Hard-coded FFT settings (clean UI)
        n_fft = 1024
        hop = 256
        st.write("FFT window size: 1024 (fixed)")
        st.write("Hop length: 256 (fixed)")

        # Only one time slider now: window start
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

    # Amplitude filtering
    with st.expander("🔊 Amplitude filtering", expanded=True):
        amp_cut = st.slider("Minimum amplitude (dB)", -120, 0, -80)
        keep_top_percent = st.slider("Keep top (%) strongest points", 1, 50, 10)

    # Display settings
    with st.expander("🎨 Display & detection", expanded=True):
        mode = st.radio("Display mode", ["Scatter", "Heatmap"], index=0)
        overlay_style = st.selectbox(
            "Call overlay style",
            ["None", "Rectangles", "Vertical bars", "Dots"],
            index=0,
        )
        detection_mode = st.selectbox(
            "Detection mode",
            ["Threshold-based", "Peak-based"],
            index=0,
        )

# =========================================================
# RIGHT COLUMN – SONOGRAM + INFO
# =========================================================
with col_plot:

    # ---------- Compute STFT on current chunk ----------
    f, t_local, Zxx = stft(y_chunk, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    t = t_local + chunk_start  # shift to global time
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)

    # Time window mask (5-second chunk)
    time_mask = (t >= chunk_start) & (t <= chunk_end)
    t_sel = t[time_mask]
    S_db_sel = S_db[:, time_mask]

    # Frequency zoom mask
    f_min = min_khz * 1000
    f_max = max_khz * 1000
    freq_mask = (f >= f_min) & (f <= f_max)
    f_sel = f[freq_mask]
    S_db_sel = S_db_sel[freq_mask, :]

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

    # ---------- Call detection ----------
    calls = []
    if overlay_style != "None":
        energy_time = S_db_sel.mean(axis=0)
        det_thresh = np.percentile(energy_time, 75)
        active = energy_time > det_thresh

        if active.any():
            idx = np.where(active)[0]
            groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for g in groups:
                t_start = t_sel[g[0]]
                t_end = t_sel[g[-1]]

                if detection_mode == "Peak-based":
                    sub = S_db_sel[:, g]
                    peak_idx = np.unravel_index(np.argmax(sub), sub.shape)
                    peak_f = f_sel[peak_idx[0]]
                else:
                    peak_f = (f_min + f_max) / 2

                calls.append(
                    {
                        "t_start": t_start,
                        "t_end": t_end,
                        "t_center": 0.5 * (t_start + t_end),
                        "f_peak": peak_f,
                    }
                )

        # Draw overlays
        for c in calls:
            if overlay_style == "Rectangles":
                fig.add_shape(
                    type="rect",
                    x0=c["t_start"],
                    x1=c["t_end"],
                    y0=f_min,
                    y1=f_max,
                    line=dict(color="cyan", width=1),
                    fillcolor="rgba(0,255,255,0.05)",
                )
            elif overlay_style == "Vertical bars":
                fig.add_shape(
                    type="line",
                    x0=c["t_start"],
                    x1=c["t_start"],
                    y0=f_min,
                    y1=f_max,
                    line=dict(color="cyan", width=2),
                )
            elif overlay_style == "Dots":
                fig.add_trace(
                    go.Scatter(
                        x=[c["t_center"]],
                        y=[c["f_peak"]],
                        mode="markers",
                        marker=dict(color="cyan", size=6),
                        showlegend=False,
                    )
                )

    # ---------- Plot ----------
    fig_container = st.container()
    fig_container.plotly_chart(fig, use_container_width=True)

    # ---------- Side info ----------
    st.markdown("### 🔍 Hover info")
    st.write("Hover over points to see time, frequency, and amplitude.")

    st.markdown("### 📡 Detected calls")
    if calls:
        for i, c in enumerate(calls, 1):
            st.write(
                f"Call {i}: {c['t_start']:.3f}–{c['t_end']:.3f} s, peak ≈ {c['f_peak']/1000:.1f} kHz"
            )
    else:
        st.write("No calls detected in this window/band with current settings.")










