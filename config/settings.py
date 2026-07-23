# Settings, thresholds, and file paths
# This file houses our mathematical thresholds and configuration constants.

# config/settings.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES_JSON_PATH = os.path.join(BASE_DIR, "config", "zones.json")

# Video Input (0 for default laptop webcam)
VIDEO_SOURCE = 0

# Gaze Gate Thresholds (Optimized for Edge/Webcam angles)
MAX_YAW_DEGREE = 35.0       # Loosened to 35 for natural head turns
MAX_PITCH_DEGREE = 35.0     # Loosened to 35 to account for looking down at laptop screens
IRIS_CENTER_THRESHOLD = 0.20 # Fine check: Max pupil offset ratio from absolute center

# Re-ID Settings
COSINE_SIMILARITY_THRESHOLD = 0.65