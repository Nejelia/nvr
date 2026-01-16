import os, io, json, yaml
from flask import send_file  # вверху файла, если ещё не импортировано
from flask import Response
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify, abort
from .storage import ensure_dirs, get_logger, list_people
from .camera import CameraWorker
from .stream import mjpeg_generator
from .face import FaceDB
from .motion import MotionDetector
import cv2
import numpy as np
from datetime import datetime

def create_app():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),'..','config.yaml'),'r',encoding='utf-8'))
    ensure_dirs(cfg)
    logger = get_logger(cfg)

    base_dir = os.path.dirname(__file__)
    proj_root = os.path.normpath(os.path.join(base_dir, ".."))

    app = Flask(
    __name__,
    template_folder=os.path.join(proj_root, "templates"),
    static_folder=os.path.join(proj_root, "static"),
    )
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

    # Cameras registry
    cameras = {}
    for c in cfg["cameras"]:
        cameras[c["id"]] = CameraWorker(cfg, c, logger)

    # Face DB helper
    face_db = FaceDB(cfg)
    face_db.load() or face_db.train()

    @app.route('/')
    def index():
        return render_template('index.html', cfg=cfg, cameras=list(cameras.keys()))

    @app.route('/stream/<int:cam_id>.mjpg')
    def stream(cam_id):
        cam = cameras.get(cam_id)
        if not cam:
            abort(404)
        return Response(mjpeg_generator(cam.get_frame, fps=cfg["video"]["fps"]),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # Control API
    @app.post('/api/start')
    def api_start():
        data = request.json or {}
        # data = { "modes": { "0": "motion", "1": "face", "2": "on_motion", "3": null } }
        modes = data.get("modes", {})
        for k,v in modes.items():
            cam = cameras.get(int(k))
            if not cam: 
                continue
            if v:
                cam.start(v)
            else:
                cam.stop()
        return jsonify({"ok": True})

    @app.post('/api/stop')
    def api_stop():
        for cam in cameras.values():
            cam.stop()
        return jsonify({"ok": True})

    # Logs
    @app.get('/events/<path:folder>/<path:fname>')
    def events_file(folder, fname):
        base = os.path.normpath(os.path.join(cfg['paths']['recordings_dir'], '..', 'events', folder))
        return send_from_directory(base, fname)

    @app.get('/logs')
    def logs_page():
        log_file = cfg["logging"]["file"]
        lines = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-1000:]
        return render_template('logs.html', lines=lines)

    # Faces management
    @app.get('/faces')
    def faces_list():
        return render_template('faces.html', faces=list_people(cfg["paths"]["faces_dir"]))

    @app.post('/faces/add')
    def faces_add():
        name = request.form.get('name','').strip()
        cam_id = request.form.get('cam_id')
        if not name:
            return redirect(url_for('faces_list'))
        cam = cameras.get(int(cam_id)) if cam_id is not None and cam_id.isdigit() else None
        imgs = []
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            data = np.frombuffer(file.read(), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            crops = face_db.detect_and_crop(bgr)
            imgs = [c for _,c in crops]
        elif cam is not None:
            frame = cam.get_frame()
            if frame is not None:
                crops = face_db.detect_and_crop(frame)
                imgs = [c for _,c in crops]
        if imgs:
            face_db.add_face_images(name, imgs)
            face_db.train()
        return redirect(url_for('faces_list'))

    @app.post('/faces/delete')
    def faces_delete():
        name = request.form.get('name','').strip()
        d = os.path.join(cfg["paths"]["faces_dir"], name)
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            # retrain
            face_db.train()
        return redirect(url_for('faces_list'))

    # Masks
    @app.get('/masks')
    def masks_page():
        return render_template('masks.html', cameras=list(cameras.keys()))

    @app.get('/roles')
    def roles_page():
        return render_template('roles.html')

    @app.post('/masks/upload')
    def masks_upload():
        cam_id = int(request.form.get('cam_id','0'))
        file = request.files.get('file')
        cam = cameras.get(cam_id)
        if not cam or not file:
            return redirect(url_for('masks_page'))
        # Expecting black/white PNG where white areas are active for motion detection.
        data = np.frombuffer(file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cam.motion.save_mask(bgr)
        return redirect(url_for('masks_page'))

    # Simple JSON status
    @app.get('/api/status')
    def api_status():
        stat = {}
        for cid, cam in cameras.items():
            stat[cid] = {"mode": cam.mode, "recording": cam.recording}
        return jsonify(stat)

    # Static files for masks to download/edit
    @app.get('/download/mask/<int:cam_id>')
    

# ... внутри create_app() ...
    @app.get('/download/mask/<int:cam_id>')
    def download_mask(cam_id):
        """
        Robust mask downloader: read file bytes and return a Response with attachment headers.
        If the mask does not exist, create a black one first.
        """
        try:
            masks_dir = cfg["paths"]["masks_dir"]
            suffix = cfg["motion"].get("mask_suffix", "_mask.png")
            fname = f"cam{cam_id}{suffix}"
            p = os.path.normpath(os.path.join(masks_dir, fname))

            os.makedirs(masks_dir, exist_ok=True)

            # If file missing — create blank mask
            if not os.path.exists(p):
                w, h = cfg["video"]["width"], cfg["video"]["height"]
                import numpy as np, cv2
                blank = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(p, blank)
                app.logger.info(f"Created blank mask for cam {cam_id} at {p}")

            # Read file bytes explicitly and return as attachment
            with open(p, "rb") as f:
                data = f.read()

            
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": f'attachment; filename="{fname}"'
            }
            return Response(data, headers=headers, status=200)
        except Exception as e:
            # Log full traceback to the app logger (and to console)
            import traceback, sys
            tb = traceback.format_exc()
            app.logger.error("Failed to serve/download mask: %s", tb)
            print("Failed to serve/download mask:", tb, file=sys.stderr)
            return ("Failed to provide mask (see server log).", 500)


    @app.get('/healthz')
    def healthz():
        return "ok"

    return app
