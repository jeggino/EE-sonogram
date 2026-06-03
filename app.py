import streamlit as st
import numpy as np
import soundfile as sf
import io
import plotly.express as px
from scipy.signal import stft

st.title("Interactive Bat Sonogram (Scatter Plot)")

uploaded_file = st.file_uploader("Upload audio", type=["wav", "flac", "mp3"])

if uploaded_file:
    # Load audio
    data, sr = sf.read(io.BytesIO(uploaded_file.read()))
    y = data.astype(float)

    # Compute STFT
    f, t, Zxx = stft(y, fs=sr, nperseg=1024, noverlap=512)
    S = np.abs(Zxx)

    # Convert to dB
    S_db = 20 * np.log10(S + 1e-12)

    # Flatten into scatter points
    T, F = np.meshgrid(t, f)
    df = {
        "time": T.flatten(),
        "freq": F.flatten(),
        "amp": S_db.flatten()
    }

    # Plotly scatter plot
    fig = px.scatter(
        df,
        x="time",
        y="freq",
        color="amp",
        color_continuous_scale="magma",
        render_mode="webgl",
        opacity=0.7,
        title="Interactive Sonogram"
    )

    fig.update_layout(
        height=600,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, 120000])  # adjust for bat frequencies
    )

    st.plotly_chart(fig, use_container_width=True)


