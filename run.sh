#!/usr/bin/env bash
set -euo pipefail

# === Raspberry Pi setup helper ===
# 1) (Recommended) System OpenCV for Pi:
#    sudo apt update
#    sudo apt install -y python3-opencv libatlas-base-dev libopenblas-dev liblapack-dev
# 2) (Optional) If you need LBPH Face Recognizer from contrib and apt OpenCV lacks it:
#    sudo pip3 install --no-cache-dir opencv-contrib-python
# 3) Web deps:
#    python3 -m pip install -r requirements.txt

# Create default folders
mkdir -p data/faces data/masks data/logs data/recordings

# Start Flask (gevent) on 0.0.0.0:8080
export FLASK_APP=app/web.py
export FLASK_ENV=production
python3 -m gevent.pywsgi -w 1 -b 0.0.0.0:8080 "app.web:create_app()"
