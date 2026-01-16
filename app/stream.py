import cv2
from flask import Response

def mjpeg_generator(get_frame_fn, fps=15):
    delay = 1.0 / max(1, fps)
    while True:
        frame = get_frame_fn()
        if frame is None:
            # send blank
            import numpy as np, time
            blank = (255 * (np.ones((240,320,3), dtype=np.uint8))).copy()
            ret, jpeg = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(jpeg) + b'\r\n')
            time.sleep(delay)
            continue
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(jpeg) + b'\r\n')
