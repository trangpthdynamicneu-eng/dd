# detector.py
from ultralytics import YOLO
import os

class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt", device=None):
        # If model file not exist, YOLO() will automatically download yolov8n.pt
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

    def detect(self, image_path, conf=0.25):
        """
        Returns list of boxes (x1,y1,x2,y2) for detected vehicles (COCO classes: car,motorcycle,bus,truck)
        """
        results = self.model(image_path, conf=conf)
        boxes = []
        for r in results:
            # r.boxes is a Boxes object; iterate
            for box in r.boxes:
                cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls[0])
                # COCO indices: car=2, motorcycle=3, bus=5, truck=7 (Ultralytics uses COCO mapping)
                if cls in (2, 3, 5, 7):
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0].numpy()
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    boxes.append((x1, y1, x2, y2))
        return boxes
