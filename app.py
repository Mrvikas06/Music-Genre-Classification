import os
import io
import time
import base64
import tempfile
import subprocess
import json
import re
import gc
import shutil
import stat
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import requests
import joblib
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import yt_dlp
import streamlit as st
import streamlit.components.v1 as components

# Ensure ffmpeg binary is discoverable in PATH across Linux/Docker/Cloud
def setup_ffmpeg():
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore
        src_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(src_exe):
            exe_dir = os.path.dirname(src_exe)
            target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            target_path = os.path.join(exe_dir, target_name)
            if not os.path.exists(target_path):
                try:
                    shutil.copy2(src_exe, target_path)
                    os.chmod(target_path, os.stat(target_path).st_mode | stat.S_IEXEC)
                except Exception:
                    pass
            if exe_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
            if os.path.exists(target_path):
                return target_path
            return src_exe
    except Exception:
        pass
    return "ffmpeg"

FFMPEG_BIN = setup_ffmpeg()


# -----------------------------------------------------------------------------
# Webpage Favicon & Branding
# -----------------------------------------------------------------------------
FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.png")
if os.path.exists(FAVICON_PATH):
    favicon_img = Image.open(FAVICON_PATH)
    with open(FAVICON_PATH, "rb") as f:
        FAVICON_B64 = base64.b64encode(f.read()).decode("utf-8")
else:
    favicon_img = "🎵"
    FAVICON_B64 = ""

# -----------------------------------------------------------------------------
# Music Genre Classification — Professional Modern Workstation
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon=favicon_img,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Favicon is cleanly managed by st.set_page_config without iframe sandbox warnings

# Helper function to prevent Markdown from interpreting indented HTML as code blocks
def render_html(html_str):
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

# Helper function to detect audio format from magic bytes
def detect_audio_extension(data: bytes) -> str:
    if data.startswith(b'RIFF'):
        return '.wav'
    elif data.startswith(b'ID3') or (len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return '.mp3'
    elif data.startswith(b'OggS'):
        return '.ogg'
    elif data.startswith(b'fLaC'):
        return '.flac'
    elif b'ftyp' in data[:24]:
        return '.m4a'
    elif data.startswith(b'\x1a\x45\xdf\xa3'):
        return '.webm'
    return '.wav'

# Helper function to check if URL is YouTube
def is_youtube_url(url: str) -> bool:
    yt_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|shorts/|live/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.search(yt_regex, url.strip()))

# Initialize Session State
for key, default in [
    ("app_state", "idle"),
    ("result", None),
    ("history", []),
    ("last_file_id", None),
    ("audio_b64", None),
    ("audio_clean_wav", None),
    ("audio_name", None),
    ("audio_size", None),
    ("audio_thumb", None),
    ("active_url", ""),
    ("uploader_key", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def clear_loaded_audio():
    st.session_state.audio_b64 = None
    st.session_state.audio_clean_wav = None
    st.session_state.audio_name = None
    st.session_state.audio_size = None
    st.session_state.audio_thumb = None
    st.session_state.last_file_id = None
    st.session_state.result = None
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.rerun()

# GTZAN Genre Knowledge Base with Icons & Modern Palette
GENRE_TAXONOMY = {
    "blues": {
        "title": "Blues",
        "icon": "🎺",
        "origin": "Mississippi Delta, USA",
        "era": "1870s – present",
        "tempo_range": "60 – 100 BPM",
        "key_marker": "12-bar blues progressions, bent-note guitar harmonics, expressive vocal calls",
        "desc": "Built on call-and-response dynamics, blue notes (flattened thirds & fifths), and expressive string bends. Spectrally concentrated in lower-mid frequencies with rich acoustic overtones.",
        "color": "#1e3264",
        "bg_tint": "rgba(30, 50, 100, 0.35)"
    },
    "classical": {
        "title": "Classical",
        "icon": "🎻",
