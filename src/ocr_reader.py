# ocr_reader.py
import easyocr
import numpy as np
import cv2

class OCRReader:
    def __init__(self, langs=['en', 'vi']):
        # Khởi tạo class OCRReader
        # easyocr.Reader là mô hình OCR dùng để nhận diện ký tự trong ảnh
        # - langs: danh sách ngôn ngữ (ở đây mặc định là tiếng Anh ['en'])
        # - gpu=False: chạy trên CPU, nếu bạn có GPU (CUDA + driver) thì có thể đổi thành True để tăng tốc
        self.reader = easyocr.Reader(langs, gpu=False)

    def read_plate(self, plate_bgr):
        if plate_bgr is None:
            return ""

        # Chuyển sang grayscale
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

        # Làm mượt ảnh để giảm nhiễu
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Dùng adaptive threshold để làm rõ ký tự
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )

        # OCR trên ảnh binary thay vì ảnh gốc
        try:
            res = self.reader.readtext(binary)
        except Exception as e:
            print("[OCR] readtext failed:", e)
            return ""

        texts = []
        for item in res:
            if len(item) == 3:
                txt = item[1]
            elif len(item) == 2:
                txt = item[0]
            else:
                txt = ""
            texts.append(txt.strip())

        return " ".join(texts)
