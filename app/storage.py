import os
from logging.handlers import RotatingFileHandler
import logging

def ensure_dirs(cfg):
    # Базовые каталоги
    for key in ["faces_dir", "masks_dir", "logs_dir", "recordings_dir"]:
        d = cfg["paths"][key]
        os.makedirs(d, exist_ok=True)

    # Каталоги медиа событий (мини-снимки/клипы) под data/events/{thumbs,clips}
    base_data = os.path.dirname(cfg["paths"]["logs_dir"])  # .../data
    events_base = os.path.join(base_data, "events")
    os.makedirs(os.path.join(events_base, "thumbs"), exist_ok=True)
    os.makedirs(os.path.join(events_base, "clips"), exist_ok=True)

def get_logger(cfg):
    os.makedirs(os.path.dirname(cfg["logging"]["file"]), exist_ok=True)
    logger = logging.getLogger("atm_cctv")
    logger.setLevel(getattr(logging, cfg["logging"]["level"]))
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        handler = RotatingFileHandler(
            cfg["logging"]["file"], maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def list_people(faces_dir):
    people = []
    if not os.path.isdir(faces_dir):
        return people
    for name in sorted(os.listdir(faces_dir)):
        path = os.path.join(faces_dir, name)
        if os.path.isdir(path):
            people.append(name)
    return people
