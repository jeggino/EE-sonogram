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

    st.write(f"Sample rate: {sr} Hz")
    st.audio(uploaded_file)

    # STFT parameters
    st.subheader("Spectrogram settings")
    n_fft = st.slider("FFT window size", 256, 4096, 1024)
    hop = st.slider("Hop length", 64, 2048, 256)

    # Compute STFT
    f, t, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)

    # Flatten into scatter points
    T, F = np.meshgrid(t, f)
    df = {
        "time": T.flatten(),
        "freq": F.flatten(),
        "amp": S_db.flatten()
    }

    # Frequency zoom
    max_freq = st.slider("Max frequency (kHz)", 20, int(sr/2000), 120)

    # Plotly scatter
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
        height=650,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, max_freq * 1000])
    )

    st.plotly_chart(fig, use_container_width=True)



