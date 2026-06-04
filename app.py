import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import stft
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from help_text import HELP_TEXT

st.set_page_config(layout="wide")
st.title("Interactive Bat Sonogram – Pro Interface")

# ---------------------------------------------------------
# Helper: restore zoom
# ---------------------------------------------------------
def restore_zoom(fig):
    zoom = st.session_state.get("zoom_state", None)
    if zoom:
        if "x" in zoom:
            fig.update_xaxes(range=zoom["x"])
        if "y" in zoom:
            fig.update_yaxes(range=zoom["y"])
    return fig


# ---------------------------------------------------------
# File upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload ultrasonic audio", type=["wav", "flac", "mp3"])

if not uploaded_file:
    st.info("Upload an ultrasonic recording to begin.")
    st.stop()

raw = uploaded_file.read()
data, sr = sf.read(io.BytesIO(raw))
if data.ndim > 1:
    data = data.mean(axis=1)
y = data.astype(float)
duration = len(y) / sr

st.write(f"Sample rate: {sr} Hz, duration: {duration:.3f} s")

# ---------------------------------------------------------
# Chunks
# ---------------------------------------------------------
chunk_size = 5.0
num_chunks = int(np.ceil(duration / chunk_size))

chunk_labels = []
for i in range(num_chunks):
    if i == 0:
        chunk_labels.append("First chunk")
    elif i == 1:
        chunk_labels.append("Second chunk")
    elif i == 2:
        chunk_labels.append("Third chunk")
    else:
        chunk_labels.append(f"{i+1} chunk")

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
col_controls, col_plot = st.columns([1, 2])

# ---------------------------------------------------------
# LEFT COLUMN – CONTROLS
# ---------------------------------------------------------
with col_controls:

    # Help dialog
    if st.button("ℹ️ How to use the app"):
        with st.dialog("App Instructions"):
            st.markdown(
                "<div style='height:70vh; overflow-y:auto; padding-right:12px;'>",
                unsafe_allow_html=True
            )
            st.markdown(HELP_TEXT)
            st.markdown("</div>", unsafe_allow_html=True)

    # Metadata
    with st.expander("📄 Project metadata", expanded=True):
        project_name = st.text_input("Project name", "")
        project_location = st.text_input("Location & date", "")
        project_species = st.text_input("Species", "")

    # Spectrogram settings
    with st.expander("⚙️ Spectrogram settings", expanded=True):

        n_fft = 1024
        hop = 256

        selected_chunk_label = st.selectbox(
            "Select chunk (5 s each)",
            chunk_labels,
            index=0,
        )
        chunk_index = chunk_labels.index(selected_chunk_label)
        chunk_start = chunk_index * chunk_size
        chunk_end = min(duration, chunk_start + chunk_size)

        st.write(f"Showing {selected_chunk_label}: {chunk_start:.2f}–{chunk_end:.2f} s")

        min_khz, max_khz = st.slider(
            "Frequency range (kHz)",
            5, int(sr / 2000),
            (15, 120)
        )

        mode = st.radio("Display mode", ["Scatter", "Heatmap"], index=0)

        if mode == "Scatter":
            point_size = st.slider("Scatter point size", 1, 10, 2)
        else:
            point_size = None

        colormap = st.selectbox(
            "Colormap",
            ["plasma", "magma", "viridis", "inferno", "cividis"],
            index=0
        )

    # Amplitude filtering
    with st.expander("🔊 Amplitude filtering", expanded=True):
        amp_cut = st.slider("Minimum amplitude (dB)", -120, 0, -80)
        keep_top_percent = st.slider("Keep top (%) strongest points", 1, 50, 10)

# ---------------------------------------------------------
# RIGHT COLUMN – SONOGRAM (ZOOM-PRESERVING VERSION)
# ---------------------------------------------------------
with col_plot:

    # Compute STFT data (this can change)
    start_sample = int(chunk_start * sr)
    end_sample = int(chunk_end * sr)
    y_chunk = y[start_sample:end_sample]

    f, t_local, Zxx = stft(y_chunk, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    t = t_local + chunk_start
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)

    f_min = min_khz * 1000
    f_max = max_khz * 1000
    freq_mask = (f >= f_min) & (f <= f_max)
    f_sel = f[freq_mask]
    S_db_sel = S_db[freq_mask, :]

    # Flatten for scatter
    T, F = np.meshgrid(t, f_sel)
    time_vals = T.flatten()
    freq_vals = F.flatten()
    amp_vals = S_db_sel.flatten()

    # Amplitude filtering
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

    # Downsample
    max_points = 200_000
    if len(time_vals) > max_points:
        idx = np.random.choice(len(time_vals), max_points, replace=False)
        time_vals = time_vals[idx]
        freq_vals = freq_vals[idx]
        amp_vals = amp_vals[idx]

    # ---------------------------------------------------------
    # FIGURE PERSISTENCE (THIS IS THE FIX)
    # ---------------------------------------------------------
    if "fig" not in st.session_state:
        st.session_state.fig = go.Figure()

    fig = st.session_state.fig

    # Clear previous traces but KEEP layout (keeps zoom)
    fig.data = []

    # Add new data
    if mode == "Scatter":
        fig.add_trace(
            go.Scattergl(
                x=time_vals,
                y=freq_vals,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=amp_vals,
                    colorscale=colormap,
                    showscale=True
                )
            )
        )
    else:
        fig.add_trace(
            go.Heatmap(
                x=t,
                y=f_sel,
                z=S_db_sel,
                colorscale=colormap,
                colorbar=dict(title="Amplitude (dB)")
            )
        )

    # Titles
    title_main = project_name if project_name else "Sonogram"
    title_sub = project_location if project_location else ""
    title_species = project_species if project_species else ""

    fig.update_layout(
        title=dict(
            text=f"<b>{title_main}</b><br><sup>{title_sub}</sup><br><sup>{title_species}</sup>",
            x=0.5,
            xanchor="center"
        ),
        height=700,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
    )

    # ---------------------------------------------------------
    # SHOW PLOT (ZOOM IS PRESERVED)
    # ---------------------------------------------------------
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True},
        key="main_plot"
    )
















