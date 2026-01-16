import cv2
import os
import numpy as np
from typing import Optional, Tuple

class FaceDB:
    def __init__(self, cfg):
        self.cfg = cfg
        self.people_dir = cfg["paths"]["faces_dir"]
        self.model_path = os.path.join(self.people_dir, "lbph_model.yml")
        self.labels_path = os.path.join(self.people_dir, "labels.json")
        self.recognizer = None
        self.labels = {}  # id->name
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + self.cfg["face"]["detection"]["cascade"])

    def _new_lbph(self):
        p = self.cfg["face"]["lbph"]
        r = cv2.face.LBPHFaceRecognizer_create(radius=p["radius"],
                                               neighbors=p["neighbors"],
                                               grid_x=p["grid_x"],
                                               grid_y=p["grid_y"])
        return r

    def train(self):
        images = []
        y = []
        label_to_id = {}
        next_id = 0
        for name in sorted(os.listdir(self.people_dir)):
            p = os.path.join(self.people_dir, name)
            if not os.path.isdir(p): 
                continue
            label_to_id.setdefault(name, next_id)
            lid = label_to_id[name]
            if lid == next_id: next_id += 1
            for fn in os.listdir(p):
                fp = os.path.join(p, fn)
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None: 
                    continue
                images.append(img)
                y.append(lid)
        if not images:
            self.recognizer = None
            self.labels = {}
            return False
        self.recognizer = self._new_lbph()
        self.recognizer.train(images, np.array(y))
        # Save
        os.makedirs(self.people_dir, exist_ok=True)
        self.recognizer.write(self.model_path)
        # invert mapping id->name
        self.labels = {lid:name for name,lid in label_to_id.items()}
        import json
        with open(self.labels_path,"w",encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
        return True

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            try:
                self.recognizer = self._new_lbph()
                self.recognizer.read(self.model_path)
                import json
                self.labels = json.load(open(self.labels_path,"r",encoding="utf-8"))
                return True
            except Exception:
                self.recognizer = None
        return False

    def add_face_images(self, person_name: str, imgs_gray):
        d = os.path.join(self.people_dir, person_name)
        os.makedirs(d, exist_ok=True)
        # Store normalized faces
        for i, face in enumerate(imgs_gray, start=1):
            face = cv2.resize(face, (200,200))
            cv2.imwrite(os.path.join(d, f"{i:04d}.png"), face)

    def detect_and_crop(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray,
                                               scaleFactor=self.cfg["face"]["detection"]["scale_factor"],
                                               minNeighbors=self.cfg["face"]["detection"]["min_neighbors"],
                                               minSize=(self.cfg["face"]["min_face_size"], self.cfg["face"]["min_face_size"]))
        crops = []
        for (x,y,w,h) in faces:
            crop = gray[y:y+h, x:x+w]
            crops.append(((x,y,w,h), cv2.resize(crop,(200,200))))
        return crops

    def recognize(self, frame_bgr) -> Tuple[Optional[str], float, Optional[tuple]]:
        if self.recognizer is None:
            return None, float("inf"), None
        crops = self.detect_and_crop(frame_bgr)
        best = (None, float("inf"), None)  # name, dist, box
        for (x,y,w,h), face in crops:
            label, conf = self.recognizer.predict(face)
            name = self.labels.get(str(label)) or self.labels.get(label)
            if conf < best[1]:
                best = (name, conf, (x,y,w,h))
        return best
