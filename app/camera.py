import cv2
import threading
import time
import os
from collections import deque

from .motion import MotionDetector
from .face import FaceDB


class CameraWorker:
    def __init__(self, cfg, cam_cfg, logger):
        self.cfg = cfg
        self.cam_id = cam_cfg["id"]
        self.device_index = cam_cfg["device_index"]
        self.logger = logger

        self.width = cfg["video"]["width"]
        self.height = cfg["video"]["height"]
        self.fps = cfg["video"]["fps"]
        self.mode = None  # 'face' | 'motion' | 'on_motion' | None

        self.cap = None
        self.thread = None
        self.stopped = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.motion = MotionDetector(cfg, self.cam_id)
        self.face_db = FaceDB(cfg)
        self.face_db.load() or self.face_db.train()

        self.recording = False
        self.writer = None

        # Буфер для стрима и предбуфер для клипов событий (~2s)
        self.buffer = deque(maxlen=int(max(1, self.fps) * 2))

        # Параметры клипов событий
        self.event_clip_active = False
        self.event_clip_writer = None
        self.event_clip_until = 0.0
        self.last_snapshot_ts = 0.0

    # ----------------------- public control -----------------------

    def start(self, mode: str):
        self.mode = mode
        if self.thread and self.thread.is_alive():
            return
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.cfg["video"].get("fourcc"):
            fourcc = cv2.VideoWriter_fourcc(*self.cfg["video"]["fourcc"])
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        self.stopped.clear()
        self.thread = threading.Thread(target=self._loop, name=f"Cam{self.cam_id}", daemon=True)
        self.thread.start()
        self.logger.info(f"Camera {self.cam_id} started in mode {self.mode}")

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self._stop_recording()
        self._finish_event_clip()
        self.thread = None
        self.logger.info(f"Camera {self.cam_id} stopped")

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    # ----------------------- recording helpers -----------------------

    def _start_recording(self):
        if self.recording:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.cfg["paths"]["recordings_dir"], f"cam{self.cam_id}_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
        self.recording = True
        self.logger.info(f"Recording started for cam {self.cam_id}: {path}")

    def _stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        if self.recording:
            self.logger.info(f"Recording stopped for cam {self.cam_id}")
        self.recording = False

    # ----------------------- event media helpers -----------------------

    def _downscale(self, frame, w=320, h=240):
        return cv2.resize(frame, (w, h))

    def _save_event_snapshot(self, frame):
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"cam{self.cam_id}_{ts}.jpg"
        out_dir = os.path.normpath(os.path.join(self.cfg["paths"]["recordings_dir"], "..", "events", "thumbs"))
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, name)
        small = self._downscale(frame, 320, 240)
        cv2.imwrite(p, small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return name  # только имя файла (UI отдаёт через /events/thumbs/<name>)

    def _start_event_clip(self):
        if self.event_clip_active:
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"cam{self.cam_id}_{ts}.avi"
        out_dir = os.path.normpath(os.path.join(self.cfg["paths"]["recordings_dir"], "..", "events", "clips"))
        os.makedirs(out_dir, exist_ok=True)
        outp = os.path.join(out_dir, name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.event_clip_writer = cv2.VideoWriter(outp, fourcc, self.fps, (320, 240))
        self.event_clip_active = True
        # выгружаем предбуфер (~2s до события)
        for frm in list(self.buffer):
            self.event_clip_writer.write(self._downscale(frm, 320, 240))
        self.event_clip_until = time.time() + 3.0  # ещё ~3s после триггера
        return name

    def _update_event_clip(self, frame):
        """Записывает кадр в клип; возвращает True, если клип завершён в этом вызове."""
        if not self.event_clip_active:
            return False
        self.event_clip_writer.write(self._downscale(frame, 320, 240))
        if time.time() > self.event_clip_until:
            self._finish_event_clip()
            return True
        return False

    def _extend_event_clip(self):
        if self.event_clip_active:
            self.event_clip_until = time.time() + 3.0

    def _finish_event_clip(self):
        if self.event_clip_writer:
            self.event_clip_writer.release()
        self.event_clip_writer = None
        self.event_clip_active = False
        self.event_clip_until = 0.0

    # ----------------------- main loop -----------------------

    def _loop(self):
        motion_cooldown = 0.0
        while not self.stopped.is_set():
            ok, frame = self.cap.read() if self.cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            triggered = False
            event_meta = None  # dict описания события для лога

            # Motion (работает в режимах motion / on_motion / face — для подстветки и клипов)
            if self.mode in ("motion", "on_motion", "face"):
                trig, boxes, _ = self.motion.detect(frame)
                if trig:
                    triggered = True
                    for (x, y, w, h) in boxes:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    event_meta = {"type": "motion", "boxes": len(boxes)}

            # Face (только в режиме face)
            if self.mode == "face":
                name, conf, box = self.face_db.recognize(frame)
                if name is not None and conf < 80.0:  # LBPH: меньше = лучше
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{name} {conf:.1f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    triggered = True
                    event_meta = {"type": "face", "name": name, "conf": float(conf)}

            # Управление записью для режима on_motion
            if self.mode == "on_motion":
                if triggered and not self.recording:
                    self._start_recording()
                    motion_cooldown = time.time() + 5.0  # минимум 5s записи после последнего движения
                elif triggered and self.recording:
                    motion_cooldown = time.time() + 5.0
                elif self.recording and time.time() > motion_cooldown:
                    self._stop_recording()

            # Сохранение клипа события и мини-снимка (дёшево по диску)
            if triggered:
                now = time.time()
                snapshot_name = None
                clip_name = None
                if now - self.last_snapshot_ts > 1.5:  # debounce ~1.5s
                    snapshot_name = self._save_event_snapshot(frame)
                    clip_name = self._start_event_clip()
                    self.last_snapshot_ts = now
                    # Логируем единоразово на триггер
                    try:
                            self.logger.info(
                            f"EVENT cam={self.cam_id} meta={event_meta} snapshot={snapshot_name} clip={clip_name}"
                        )
                    except Exception:
                        # на всякий случай не роняем поток
                        pass
                else:
                    # если событие продолжается — удлиним клип
                    self._extend_event_clip()

            # Если клип активен — пишем кадры, завершаем по таймеру
            if self.event_clip_active:
                self._update_event_clip(frame)

            # Если идёт запись «on_motion» — пишем полноразмерный AVI
            if self.recording and self.writer is not None:
                self.writer.write(frame)

            # Обновляем последний кадр и предбуфер
            with self.frame_lock:
                self.latest_frame = frame
                self.buffer.append(frame)

            # лёгкая пауза
            time.sleep(0.001)

        # Cleanup
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self._stop_recording()
        self._finish_event_clip()
