HELP_TEXT = """
# 🦇 How to Use the Bat Sonogram App

### 1️⃣ Uploading a File
- Click **Upload ultrasonic audio** and select a `.wav`, `.flac`, or `.mp3` file.
- The app automatically reads the file and shows its duration and sample rate.
- Long recordings are divided into **5‑second chunks** for faster processing.

### 2️⃣ Navigating Chunks
- Use the **Select chunk (5 s each)** dropdown to explore different parts of the recording.
- Each chunk represents a 5‑second segment of your audio.
- The sonogram updates automatically when you switch chunks.

### 3️⃣ Spectrogram Settings
- **Frequency range (kHz)**: Adjust the range of frequencies displayed using the range slider.
- **Display mode**: Choose between:
  - *Scatter* — individual points colored by amplitude.
  - *Heatmap* — continuous color representation of amplitude.
- **Scatter point size**: Visible only when Scatter mode is selected; controls dot size.
- **Colormap**: Choose the color palette (default: *plasma*).

### 4️⃣ Amplitude Filtering
- **Minimum amplitude (dB)**: Filters out weaker signals below the threshold.
- **Keep top (%) strongest points**: Keeps only the most intense points for clarity.

### 5️⃣ Metadata
- In **Project metadata**, enter:
  - *Project name* → becomes the main title (H1)
  - *Location & date* → becomes subtitle (H2)
  - *Species* → becomes secondary subtitle (H3)
- These appear above the sonogram for easy documentation.

### 6️⃣ Viewing the Sonogram
- The sonogram shows time (horizontal axis) vs. frequency (vertical axis).
- Color intensity represents amplitude (signal strength).
- Use the controls to zoom, filter, and adjust visualization style.

### 7️⃣ Tips
- For best results, use recordings sampled at ≥192 kHz.
- Keep amplitude filtering moderate to avoid losing faint calls.
- Use consistent metadata for field projects.

Enjoy exploring your bat recordings!
"""
