from ultralytics import YOLO
import easyocr
import tensorflow as tf

print("✅ TensorFlow version:", tf.__version__)
print("✅ YOLOv8 test:")
model = YOLO("models/yolov8n.pt")
results = model("test.jpg")
print("Detections:", results[0].boxes.xyxy)

print("✅ EasyOCR test:")
reader = easyocr.Reader(['en'])
res = reader.readtext("test.jpg")
print("OCR results:", res[:3])  # in thử 3 kết quả đầu
