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
        "origin": "Western Europe",
        "era": "9th century – present",
        "tempo_range": "40 – 160 BPM",
        "key_marker": "Dynamic range, uncompressed acoustic orchestration, intricate harmonic forms",
        "desc": "Defined by expansive dynamic headroom, complex orchestral counterpoint, and acoustic timbral purity. Exhibits premier spectral centroid separation across strings and orchestral brass.",
        "color": "#7d4b32",
        "bg_tint": "rgba(125, 75, 50, 0.35)"
    },
    "country": {
        "title": "Country",
        "icon": "🪕",
        "origin": "Southern USA",
        "era": "1920s – present",
        "tempo_range": "80 – 130 BPM",
        "key_marker": "Acoustic strings (banjo, fiddle, pedal steel), diatonic storytelling form",
        "desc": "Harmonically anchored by diatonic folk progressions and organic acoustic strings. Feature vectors cluster between folk and blues with prominent vocal-forward master compression.",
        "color": "#d84000",
        "bg_tint": "rgba(216, 64, 0, 0.35)"
    },
    "disco": {
        "title": "Disco",
        "icon": "🪩",
        "origin": "New York, USA",
        "era": "1970s – 1980s",
        "tempo_range": "115 – 130 BPM",
        "key_marker": "Four-on-the-floor kick, syncopated electric bass, orchestral string accents",
        "desc": "Features metronomic 4/4 timing with open hi-hats on offbeats and syncopated slap bass. Spectral profiles reveal high low-end energy paired with shimmering high-frequency string sections.",
        "color": "#dc148c",
        "bg_tint": "rgba(220, 20, 140, 0.35)"
    },
    "hiphop": {
        "title": "Hip-Hop",
        "icon": "🎤",
        "origin": "Bronx, New York, USA",
        "era": "1970s – present",
        "tempo_range": "80 – 105 BPM",
        "key_marker": "Sub-bass 808s, transient breakbeats, rhythmic speech cadences",
        "desc": "Anchored by programmed drum break transients, punchy 808 sub-bass, and syncopated vocal delivery. High energy density concentrated below 80 Hz with sparse mid-range harmonic density.",
        "color": "#ba5d07",
        "bg_tint": "rgba(186, 93, 7, 0.35)"
    },
    "jazz": {
        "title": "Jazz",
        "icon": "🎷",
        "origin": "New Orleans, USA",
        "era": "1900s – present",
        "tempo_range": "70 – 240 BPM",
        "key_marker": "Extended chord voicings (7ths, 9ths, 13ths), swing feel, modal improvisation",
        "desc": "Displays intricate harmonic voice leading, polyrhythmic swing, and modal interchange. Chroma feature vectors exhibit high entropy due to frequent key modulation and extended chords.",
        "color": "#477d95",
        "bg_tint": "rgba(71, 125, 149, 0.35)"
    },
    "metal": {
        "title": "Metal",
        "icon": "⚡",
        "origin": "UK / USA",
        "era": "Late 1960s – present",
        "tempo_range": "100 – 220 BPM",
        "key_marker": "High-gain distorted guitars, double-bass transients, aggressive vocal delivery",
        "desc": "Characterized by saturated harmonic overdrive, rapid double-kick percussion, and dense power riffs. Spectral rolloff and zero-crossing rates remain elevated across all frequency registers.",
        "color": "#503750",
        "bg_tint": "rgba(80, 55, 80, 0.4)"
    },
    "pop": {
        "title": "Pop",
        "icon": "📻",
        "origin": "USA / UK",
        "era": "1950s – present",
        "tempo_range": "100 – 128 BPM",
        "key_marker": "Compressed dynamic range, melodic hook symmetry, radio-mastered vocal curve",
        "desc": "Engineered for instantaneous melodic retention, verse-chorus symmetry, and controlled dynamic range. Centroid and bandwidth cluster tightly due to commercial multi-band mastering.",
        "color": "#8d67ab",
        "bg_tint": "rgba(141, 103, 171, 0.35)"
    },
    "reggae": {
        "title": "Reggae",
        "icon": "🌴",
        "origin": "Jamaica",
        "era": "1960s – present",
        "tempo_range": "60 – 90 BPM",
        "key_marker": "Offbeat guitar chop ('skank'), deep bass guitar root notes, relaxed tempo",
        "desc": "Distinguished by staccato offbeat guitar and piano accents on beats two and four, supported by heavy bass guitar fundamentals and relaxed half-time drumming.",
        "color": "#006450",
        "bg_tint": "rgba(0, 100, 80, 0.35)"
    },
    "rock": {
        "title": "Rock",
        "icon": "🎸",
        "origin": "USA / UK",
        "era": "1950s – present",
        "tempo_range": "100 – 150 BPM",
        "key_marker": "Overdriven electric guitar riffs, snare drum backbeat, driving bassline",
        "desc": "Centers on electric guitars, bass, and acoustic drum kits. Occupies an aggressive mid-frequency register with high-energy pick transients and steady backbeat snare hits.",
        "color": "#e91429",
        "bg_tint": "rgba(233, 20, 41, 0.35)"
    },
}

# -----------------------------------------------------------------------------
# Global Styling (106% Zoom Scale, Mobile-First Layout, Dark Ambient Glow)
# -----------------------------------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --spotify-green: #1ed760;
        --spotify-green-hover: #1fdf64;
        --bg-black: #121212;
        --surface-card: #181818;
        --surface-hover: #222222;
        --surface-card-elevated: #202020;
        --text-white: #ffffff;
        --text-silver: #b3b3b3;
        --text-muted: #7c7c7c;
        --border-subtle: #282828;
        --border-input: #444444;

        color-scheme: dark;
        accent-color: #1ed760;
        scrollbar-color: #535353 #121212;
    }

    html, body {
        color-scheme: dark;
        scroll-behavior: smooth;
        zoom: 1.06; /* 106% comfortable desktop zoom scale */
    }

    /* WebKit Scrollbar */
    @supports not (scrollbar-color: auto) {
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-thumb { background: #535353; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #7c7c7c; }
        ::-webkit-scrollbar-track { background: #121212; }
    }

    /* Modern Web Guidance: Scroll-driven animations with feature detection */
    @media (prefers-reduced-motion: no-preference) {
        @supports ((animation-timeline: view()) and (animation-range: entry)) {
            @keyframes card-fade-up {
                from { opacity: 0; transform: translateY(18px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .scroll-reveal {
                animation: card-fade-up auto linear backwards;
                animation-timeline: view();
                animation-range: entry 5% cover 25%;
            }
        }
    }

    #MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

    /* Page Container */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 5rem !important;
        padding-left: clamp(1rem, 3.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3.5vw, 2.5rem) !important;
        max-width: 1300px !important;
        margin: 0 auto !important;
    }

    body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #121212 !important;
        background-image: radial-gradient(circle at 50% 0%, #1a1a1a 0%, #121212 65%) !important;
        background-attachment: fixed !important;
        color: #b3b3b3 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .font-mono, code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Segmented Pill Tabs (Universal across all Streamlit versions) */
    [data-testid="stTabs"] {
        margin-top: 8px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"],
    [data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] > div[role="tablist"] {
        gap: 8px !important;
        background: #181818 !important;
        padding: 6px !important;
        border-radius: 9999px !important;
        border: 1px solid #282828 !important;
        margin-bottom: 30px !important;
        display: inline-flex !important;
        width: fit-content !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"],
    [data-testid="stTabs"] button[role="tab"],
    [data-testid="stTabs"] button,
    button[data-testid="stTab"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 9999px !important;
        color: #b3b3b3 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        letter-spacing: -0.01em !important;
        transition: all 180ms ease !important;
        white-space: nowrap !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover,
    [data-testid="stTabs"] button[role="tab"]:hover,
    [data-testid="stTabs"] button:hover,
    button[data-testid="stTab"]:hover {
        color: #ffffff !important;
        background: #242424 !important;
    }
    /* Active Tab Solid Green Pill (Strictly scoped to tablist bar) */
    [data-testid="stTabs"] [role="tablist"] [data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tablist"] button[role="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"],
    button[data-testid="stTab"][aria-selected="true"] {
        background: #1ed760 !important;
        background-color: #1ed760 !important;
        color: #000000 !important;
        border-radius: 9999px !important;
        box-shadow: 0 2px 10px rgba(30, 215, 96, 0.4) !important;
    }
    [data-testid="stTabs"] [role="tablist"] [data-baseweb="tab"][aria-selected="true"] *,
    [data-testid="stTabs"] [role="tablist"] button[role="tab"][aria-selected="true"] *,
    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] *,
    button[data-testid="stTab"][aria-selected="true"] * {
        color: #000000 !important;
        font-weight: 850 !important;
    }
    /* Suppress default underline / tab highlight */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"],
    [data-testid="stTabs"] [data-testid="stTabHighlight"],
    [data-testid="stTabHighlight"],
    [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] hr {
        display: none !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
        border: none !important;
    }

    /* File Uploader (Universal across all Streamlit versions) */
    [data-testid="stFileUploader"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploader"] section {
        background: #181818 !important;
        border: 1px dashed #444444 !important;
        border-radius: 12px !important;
        padding: 28px 20px !important;
        transition: all 200ms ease !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploadDropzone"]:hover,
    [data-testid="stFileUploader"] section:hover {
        background: #1e1e1e !important;
        border-color: #1ed760 !important;
    }
    /* Browse Files Button (Strictly scoped to dropzone) */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploader"] section button {
        background: #222222 !important;
        background-color: #222222 !important;
        border: 1px solid #7c7c7c !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        border-radius: 9999px !important;
        padding: 10px 26px !important;
        transition: all 150ms ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploadDropzone"] button *,
    [data-testid="stFileUploader"] section button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploadDropzone"] button:hover,
    [data-testid="stFileUploader"] section button:hover {
        background: #2a2a2a !important;
        border-color: #1ed760 !important;
        color: #1ed760 !important;
        transform: scale(1.02) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover *,
    [data-testid="stFileUploadDropzone"] button:hover *,
    [data-testid="stFileUploader"] section button:hover * {
        color: #1ed760 !important;
        fill: #1ed760 !important;
    }

    /* File Uploader File Card & Delete Cross (✕) Button */
    [data-testid="stFileUploaderFile"] {
        background: #181818 !important;
        border: 1px solid #282828 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        margin-top: 10px !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] button,
    [data-testid="stFileUploaderFile"] button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        max-width: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #b3b3b3 !important;
        cursor: pointer !important;
        outline: none !important;
        box-shadow: none !important;
        transition: all 160ms ease !important;
    }
    /* Preserve transparent SVG bounding box to prevent solid grey square */
    [data-testid="stFileUploaderDeleteBtn"] button svg,
    [data-testid="stFileUploaderFile"] button svg {
        width: 18px !important;
        height: 18px !important;
        color: #b3b3b3 !important;
        background: transparent !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] button svg path[fill="none"],
    [data-testid="stFileUploaderFile"] button svg path[fill="none"] {
        fill: none !important;
    }

    [data-testid="stFileUploaderDeleteBtn"] button:hover,
    [data-testid="stFileUploaderFile"] button:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: rgba(239, 68, 68, 0.6) !important;
        color: #ef4444 !important;
        transform: scale(1.08) !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] button:hover svg,
    [data-testid="stFileUploaderFile"] button:hover svg {
        color: #ef4444 !important;
    }

    /* Text Input (Unified Single-Border Pill) */
    [data-testid="stTextInput"] {
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stTextInput"] > div,
    [data-testid="stTextInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] [data-baseweb="input"] {
        background: #181818 !important;
        border: 1px solid #383838 !important;
        border-radius: 9999px !important;
        box-shadow: none !important;
        outline: none !important;
        overflow: hidden !important;
        transition: all 180ms ease !important;
    }
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stTextInput"] [data-baseweb="input"]:hover {
        border-color: #1ed760 !important;
        box-shadow: 0 0 0 2px rgba(30, 215, 96, 0.35) !important;
    }
    [data-testid="stTextInput"] input {
        background: transparent !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 14px 22px !important;
        font-size: 14.5px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] input:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* CTA Pill Buttons */
    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        background: #1ed760 !important;
        border: none !important;
        color: #000000 !important;
        font-size: 14.5px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        border-radius: 9999px !important;
        padding: 13px 28px !important;
        min-height: 48px !important;
        height: 48px !important;
        margin-top: 0 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
        transition: all 160ms ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stButton"] > button:hover:not(:disabled),
    [data-testid="stFormSubmitButton"] > button:hover:not(:disabled) {
        background: #1fdf64 !important;
        transform: scale(1.02) !important;
        box-shadow: 0 10px 28px rgba(30, 215, 96, 0.4) !important;
    }
    [data-testid="stButton"] > button:active:not(:disabled),
    [data-testid="stFormSubmitButton"] > button:active:not(:disabled) {
        transform: scale(0.99) !important;
    }
    [data-testid="stButton"] > button:disabled,
    [data-testid="stFormSubmitButton"] > button:disabled {
        opacity: 0.35 !important;
        background: #242424 !important;
        color: #7c7c7c !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    /* Premium Secondary / Clear / Close Action Pill Buttons */
    button[kind="secondary"],
    [data-testid="stButton"] > button[kind="secondary"],
    [data-testid="stFormSubmitButton"] > button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background: #181818 !important;
        background-color: #181818 !important;
        color: #e4e4e7 !important;
        border: 1px solid #333333 !important;
        border-radius: 9999px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        font-size: 12.5px !important;
        font-weight: 750 !important;
        letter-spacing: 0.8px !important;
        padding: 7px 18px !important;
        min-height: 38px !important;
        height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 180ms ease !important;
    }
    [data-testid="stFormSubmitButton"] > button[kind="secondary"] {
        min-height: 48px !important;
        height: 48px !important;
    }
    button[kind="secondary"]:hover:not(:disabled),
    [data-testid="stButton"] > button[kind="secondary"]:hover:not(:disabled),
    [data-testid="stFormSubmitButton"] > button[kind="secondary"]:hover:not(:disabled),
    [data-testid="baseButton-secondary"]:hover:not(:disabled) {
        background: rgba(239, 68, 68, 0.15) !important;
        background-color: rgba(239, 68, 68, 0.15) !important;
        color: #ff6b6b !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.25) !important;
        transform: scale(1.02) !important;
    }

    /* Seamless Borderless Form Container */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Ultra-Premium Segmented Pill Switcher for Radio */
    [data-testid="stRadio"] {
        margin: 8px 0 16px 0 !important;
    }
    [data-testid="stRadio"] [role="radiogroup"],
    [data-testid="stRadioHorizontal"] [role="radiogroup"] {
        gap: 6px !important;
        background: #141414 !important;
        padding: 5px !important;
        border-radius: 9999px !important;
        border: 1px solid #282828 !important;
        display: inline-flex !important;
        align-items: center !important;
        width: fit-content !important;
        max-width: 100% !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.6), 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stRadio"] label,
    [data-testid="stRadioHorizontal"] label {
        color: #a1a1aa !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 8px 20px !important;
        border-radius: 9999px !important;
        cursor: pointer !important;
        white-space: nowrap !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        transition: all 180ms ease !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        margin: 0 !important;
    }
    [data-testid="stRadio"] label:hover,
    [data-testid="stRadioHorizontal"] label:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    /* Completely hide ugly browser/baseweb circular radio dots */
    [data-testid="stRadio"] label > div[aria-hidden="true"],
    [data-testid="stRadio"] label > div:first-child,
    [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }

    /* Active Segment: Sleek elevated dark pill with clean white text & subtle highlight */
    [data-testid="stRadio"] label:has(input:checked),
    [data-testid="stRadio"] label[aria-checked="true"] {
        background: #252525 !important;
        background-color: #252525 !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45) !important;
    }

    /* Strip all child backgrounds, eliminating any solid green block behind text */
    [data-testid="stRadio"] label *,
    [data-testid="stRadio"] label [data-testid="stMarkdownContainer"],
    [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] *,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stRadio"] label:has(input:checked) *,
    [data-testid="stRadio"] label[aria-checked="true"] * {
        color: #ffffff !important;
        font-weight: 750 !important;
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Mobile Responsive Overrides */
    @media (max-width: 768px) {
        html, body { zoom: 1.0; }
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            border-radius: 12px !important;
            width: 100% !important;
            padding: 4px !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 12px !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }

    iframe { border: none !important; }
</style>
""")

# -----------------------------------------------------------------------------
# Model & Feature Extractor
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def get_ffmpeg_cmd():
    return setup_ffmpeg()

@st.cache_resource
def load_model_artifacts():
    mp = os.path.join(MODEL_DIR, "champion_genre_classifier.pkl")
    if not os.path.exists(mp):
        mp = os.path.join(MODEL_DIR, "genre_classifier.pkl")
    sp = os.path.join(MODEL_DIR, "scaler.pkl")
    ep = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if not all(os.path.exists(p) for p in [mp, sp, ep]):
        return None, None, None
    return joblib.load(mp), joblib.load(sp), joblib.load(ep)

def prepare_audio_for_player_and_model(raw_bytes: bytes, ext: str = ".wav"):
    """
    Normalizes audio into:
    1) clean_wav: 30-second 22,050 Hz 16-bit mono WAV for Librosa feature extraction (~1.3MB)
       (GTZAN dataset and models are trained on 30s clips)
    2) preview_mp3: Full-length track (up to 10 min) for complete playback in the interactive player
    """
    in_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    in_file.write(raw_bytes)
    in_file.close()

    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    out_wav.close()
    out_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    out_mp3.close()

    clean_wav = None
    preview_mp3 = None
    ffmpeg_bin = get_ffmpeg_cmd()

    try:
        # Standardized 30-second WAV for ML model inference
        cmd_wav = [
            ffmpeg_bin, '-nostdin', '-y', '-i', in_file.name,
            '-t', '30', '-ar', '22050', '-ac', '1',
            '-c:a', 'pcm_s16le', out_wav.name
        ]
        p1 = subprocess.run(cmd_wav, stdin=subprocess.DEVNULL, capture_output=True, timeout=25)
        if p1.returncode == 0 and os.path.exists(out_wav.name) and os.path.getsize(out_wav.name) > 0:
            with open(out_wav.name, 'rb') as f:
                clean_wav = f.read()

        # Full track for the player: if already MP3 and reasonably sized, keep original; otherwise compress to 128k MP3
        if ext.lower() == ".mp3" and len(raw_bytes) <= 15 * 1024 * 1024:
            preview_mp3 = raw_bytes
        else:
            cmd_mp3 = [
                ffmpeg_bin, '-nostdin', '-y', '-i', in_file.name,
                '-t', '600', '-ar', '44100', '-ac', '2',
                '-b:a', '128k', out_mp3.name
            ]
            p2 = subprocess.run(cmd_mp3, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
            if p2.returncode == 0 and os.path.exists(out_mp3.name) and os.path.getsize(out_mp3.name) > 0:
                with open(out_mp3.name, 'rb') as f:
                    preview_mp3 = f.read()
    except Exception:
        pass
    finally:
        for p in [in_file.name, out_wav.name, out_mp3.name]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

    # Robust fallback for clean_wav if ffmpeg is unavailable
    if clean_wav is None:
        try:
            sig, sr_in = None, 22050
            try:
                sig, sr_in = sf.read(io.BytesIO(raw_bytes), dtype='float32')
            except Exception:
                sig, sr_in = librosa.load(io.BytesIO(raw_bytes), sr=22050, duration=30, mono=True)
            if sig is not None:
                if len(sig.shape) > 1:
                    sig = np.mean(sig, axis=1)
                if sr_in != 22050:
                    sig = librosa.resample(sig, orig_sr=sr_in, target_sr=22050)
                max_len = 30 * 22050
                if len(sig) > max_len:
                    sig = sig[:max_len]
                elif len(sig) < max_len:
                    sig = np.pad(sig, (0, max_len - len(sig)))
                buf = io.BytesIO()
                sf.write(buf, sig.astype(np.float32), 22050, format='WAV', subtype='PCM_16')
                clean_wav = buf.getvalue()
        except Exception:
            clean_wav = raw_bytes

    if preview_mp3 is None:
        preview_mp3 = raw_bytes if len(raw_bytes) <= 15 * 1024 * 1024 else (clean_wav or raw_bytes)

    return clean_wav, preview_mp3

def process_audio(audio_bytes, progress_callback=None):
    signal = None
    sr = 22050
    ffmpeg_bin = get_ffmpeg_cmd()

    if progress_callback:
        progress_callback("Decoding & standardizing 30-second waveform...")

    # 1. Fast in-memory decoding via soundfile if already WAV
    try:
        signal, file_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        if file_sr != 22050:
            signal = librosa.resample(signal, orig_sr=file_sr, target_sr=22050)
        sr = 22050
    except Exception:
        signal = None

    # 2. Decode via ffmpeg if signal is not yet loaded
    if signal is None or len(signal) == 0:
        ext = detect_audio_extension(audio_bytes)
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        in_tmp.write(audio_bytes)
        in_tmp.close()
        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        out_wav.close()
        try:
            cmd = [
                ffmpeg_bin, '-nostdin', '-y',
                '-i', in_tmp.name,
                '-t', '30',
                '-ar', '22050',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                out_wav.name
            ]
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=25)
            if res.returncode == 0 and os.path.exists(out_wav.name) and os.path.getsize(out_wav.name) > 0:
                signal, sr = sf.read(out_wav.name, dtype='float32')
        except Exception:
            pass
        finally:
            for p in [in_tmp.name, out_wav.name]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass

    # 3. Fallback to librosa.load if still None
    if signal is None or len(signal) == 0:
        ext = detect_audio_extension(audio_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            signal, sr = librosa.load(tmp, sr=22050, duration=30, mono=True)
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass

    if signal is None or len(signal) == 0:
        raise ValueError("Could not decode audio stream into waveform. Please ensure the file is an uncorrupted WAV or MP3.")

    # Enforce standard 30-second duration at 22,050 Hz (exact 661,500 samples)
    max_len = 30 * sr
    if len(signal) > max_len:
        signal = signal[:max_len]
    elif len(signal) < max_len:
        signal = np.pad(signal, (0, max_len - len(signal)))
    signal = np.ascontiguousarray(signal, dtype=np.float32)

    if progress_callback:
        progress_callback("Extracting spectral dynamics (STFT, Centroid, Rolloff, Chroma)...")

    # Time-domain features
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)

    # Compute STFT once and reuse magnitude & power spectrograms to minimize memory & CPU
    D = librosa.stft(signal, n_fft=2048, hop_length=512)
    S = np.abs(D)
    S2 = S ** 2

    # Spectral features computed directly from S / S2
    chroma_stft = librosa.feature.chroma_stft(S=S2, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)

    # Mel-spectrogram & 20 MFCCs computed directly from S2
    mel_spec = librosa.feature.melspectrogram(S=S2, sr=sr)
    log_mel = librosa.power_to_db(mel_spec)
    mfccs_full = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=20)
    del mel_spec, log_mel, S2

    if progress_callback:
        progress_callback("Separating harmonic/percussive timbre & tempo estimation...")

    # HPSS decomposition with direct memory freeing
    H, P = librosa.decompose.hpss(D, margin=1.0)
    del D
    harmony = librosa.istft(H, length=len(signal))
    del H
    perceptr = librosa.istft(P, length=len(signal))
    del P
    del S
    gc.collect()

    tempo_res, _ = librosa.beat.beat_track(y=signal, sr=sr)
    est_tempo = float(tempo_res) if np.isscalar(tempo_res) else float(tempo_res[0])

    feats = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spec_cent), np.var(spec_cent),
        np.mean(spec_bw), np.var(spec_bw),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zcr), np.var(zcr),
        np.mean(harmony), np.var(harmony),
        np.mean(perceptr), np.var(perceptr),
        est_tempo,
    ]
    for i in range(20):
        feats.append(np.mean(mfccs_full[i]))
        feats.append(np.var(mfccs_full[i]))

    mfcc_13 = [float(np.mean(mfccs_full[i])) for i in range(13)]
    mean_cent = float(np.mean(spec_cent))
    mean_roll = float(np.mean(rolloff))
    mean_zcr = float(np.mean(zcr))

    if progress_callback:
        progress_callback("Evaluating SVM RBF Kernel classification...")

    return np.array(feats, dtype=np.float32).reshape(1, -1), mfcc_13, mean_cent, mean_roll, mean_zcr, est_tempo

# Extract ID3 Embedded Album Artwork from Audio Bytes
def extract_id3_image(data: bytes):
