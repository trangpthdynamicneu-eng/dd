# detector.py
from ultralytics import YOLO
import os

class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt", device=None):
        # Khởi tạo class VehicleDetector dùng YOLOv8 để phát hiện phương tiện
        # - model_path: đường dẫn tới file model YOLOv8 (ở đây mặc định là yolov8n.pt - bản nhỏ nhất)
        #   Nếu file chưa có, YOLO() sẽ tự động tải từ Ultralytics về
        # - device: có thể set "cuda" để chạy GPU hoặc "cpu"
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)  # Chuyển model sang GPU/CPU tuỳ theo device được truyền vào

    def detect(self, image_path, conf=0.25):
        """
        Hàm detect: dùng YOLOv8 phát hiện phương tiện giao thông
        Trả về danh sách toạ độ bounding box (x1, y1, x2, y2)
        conf=0.25: ngưỡng độ tin cậy (confidence threshold)
        """

        # Chạy YOLO predict trên ảnh
        results = self.model(image_path, conf=conf)

        boxes = []
        for r in results:
            # r.boxes: đối tượng chứa tất cả bounding boxes được dự đoán
            for box in r.boxes:
                # Lấy nhãn class của box
                # box.cls là tensor -> cần chuyển sang numpy rồi sang int
                cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls[0])

                # YOLOv8 sử dụng dataset COCO với mapping:
                # car = 2, motorcycle = 3, bus = 5, truck = 7
                if cls in (2, 3, 5, 7):
                    # Lấy toạ độ (x1, y1, x2, y2) của box
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0].numpy()
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    boxes.append((x1, y1, x2, y2))

        return boxes
