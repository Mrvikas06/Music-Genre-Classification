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
