#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

venv_dir="${VENV_DIR:-.venv}"
requirements_file="requirements.txt"

if [[ ! -d "$venv_dir" ]]; then
  python3 -m venv "$venv_dir"
fi

# shellcheck disable=SC1091
source "$venv_dir/bin/activate"

if [[ ! -f "$requirements_file" ]]; then
  echo "requirements file not found: $requirements_file" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install --upgrade -r "$requirements_file"

# Create default folders
mkdir -p data/faces data/masks data/logs data/recordings

# Start Flask (gevent) on 0.0.0.0:8080
export FLASK_APP=app/web.py
export FLASK_ENV=production
python -m gevent.pywsgi -w 1 -b 0.0.0.0:8080 "app.web:create_app()"
