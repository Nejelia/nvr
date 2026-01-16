import os, yaml, traceback, sys
try:
    cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    masks_dir = cfg["paths"]["masks_dir"]
    suffix = cfg["motion"].get("mask_suffix", "_mask.png")
    fname = f"cam0{suffix}"
    p = os.path.normpath(os.path.join(masks_dir, fname))
    print("CONFIG masks_dir:", masks_dir)
    print("ABS PATH:", os.path.abspath(masks_dir))
    print("TARGET FILE:", p)
    print("DIR EXISTS:", os.path.isdir(masks_dir))
    if os.path.isdir(masks_dir):
        print("DIR LISTING:", sorted(os.listdir(masks_dir)))
    else:
        print("DIR will be created.")
        os.makedirs(masks_dir, exist_ok=True)

    # Проверка прав записи
    can_write = True
    try:
        testf = os.path.join(masks_dir, ".__write_test.tmp")
        with open(testf, "wb") as f:
            f.write(b"ok")
        os.remove(testf)
        print("Write test: OK")
    except Exception as e:
        can_write = False
        print("Write test: FAILED:", repr(e))

    import numpy as np
    print("NumPy version:", np.__version__)
    import cv2
    print("OpenCV version:", cv2.__version__)

    h = cfg["video"]["height"]
    w = cfg["video"]["width"]

    blank = np.zeros((h, w), dtype=np.uint8)

    ok = cv2.imwrite(p, blank)
    print("cv2.imwrite returned:", ok)
    if ok and os.path.exists(p):
        print("File created. size:", os.path.getsize(p))
    else:
        print("File not created or size 0.")

except Exception as ex:
    print("EXCEPTION during test:")
    traceback.print_exc(file=sys.stdout)
