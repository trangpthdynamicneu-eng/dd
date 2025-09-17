# ocr_reader.py
import easyocr
import numpy as np
import cv2

class OCRReader:
    def __init__(self, langs=['en']):
        # quiet: set verbose False inside easyocr if needed
        self.reader = easyocr.Reader(langs, gpu=False)  # set gpu=True if you have GPU and drivers

    def read_plate(self, plate_bgr):
        if plate_bgr is None:
            return ""
        plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        # easyocr readtext returns list of (bbox, text, conf)
        try:
            res = self.reader.readtext(plate_rgb)
        except Exception as e:
            print("[OCR] readtext failed:", e)
            return ""
        texts = []
        for item in res:
            # item could be (bbox, text, conf) or (text, conf) depending on detail flag
            if len(item) == 3:
                txt = item[1]
            elif len(item) == 2:
                txt = item[0]
            else:
                txt = ""
            texts.append(txt)
        return " ".join(texts)
