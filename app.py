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
# Replace with your image path or URL
IMAGE_URL = "data:image/jpeg;base64,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"

st.markdown(
    f"""
    <style>
        /* Remove Streamlit default header background */
        header[data-testid="stHeader"] {{
            background: none;
        }}

        /* Add your background image */
        header[data-testid="stHeader"]::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 60px; /* adjust height */
            background-image: url("{IMAGE_URL}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            z-index: 0;
        }}

        /* Keep menu buttons clickable */
        header[data-testid="stHeader"] > div {{
            position: relative;
            z-index: 1;
        }}
    </style>
    """,
    unsafe_allow_html=True
)




# ---------------------------------------------------------
# Helper: restore zoom
# ---------------------------------------------------------

@st.dialog(" ")
def helper():
    st.markdown(HELP_TEXT)



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
        helper()


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
            point_size = st.slider("Scatter point size", 1, 10, 4)
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
        keep_top_percent = st.slider("Keep top (%) strongest points", 1, 70, 50)

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
                ),
                hovertemplate=(
                    "Time: %{x:.3f} s<br>"
                    "Frequency: %{y:.1f} Hz<br>"
                    "Amplitude: %{marker.color:.2f}<extra></extra>"
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
















