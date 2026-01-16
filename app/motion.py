import cv2
import numpy as np
import os

class MotionDetector:
    def __init__(self, cfg, cam_id: int):
        self.cfg = cfg
        self.cam_id = cam_id
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=cfg["motion"]["history"],
                                                          varThreshold=cfg["motion"]["var_threshold"],
                                                          detectShadows=cfg["motion"]["detect_shadows"])
        self.mask = self._load_mask()

    def _mask_path(self):
        suffix = self.cfg["motion"]["mask_suffix"]
        return os.path.join(self.cfg["paths"]["masks_dir"], f"cam{self.cam_id}{suffix}")

    def _load_mask(self):
        p = self._mask_path()
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1]
        return None

    def save_mask(self, mask_img_bgr):
        # mask should be white regions = active detection areas
        gray = cv2.cvtColor(mask_img_bgr, cv2.COLOR_BGR2GRAY)
        binm = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(self._mask_path(), binm)
        self.mask = binm

    def detect(self, frame):
        fg = self.backsub.apply(frame)
        if self.mask is not None and self.mask.shape[:2] == fg.shape[:2]:
            fg = cv2.bitwise_and(fg, self.mask)
        th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        th = cv2.dilate(th, None, iterations=self.cfg["motion"]["dilate_iterations"])
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) < self.cfg["motion"]["min_contour_area"]:
                continue
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,w,h))
        triggered = len(boxes) > 0
        return triggered, boxes, th
