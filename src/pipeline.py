# pipeline.py
import cv2
import os
from detector import VehicleDetector
from wpod_net import WPODProcessor
from ocr_reader import OCRReader

def draw_boxes_and_plate(img, box, plate_img, text, idx):
    x1,y1,x2,y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    # place text
    cv2.putText(img, f"{idx}:{text}", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # show plate at right side
    if plate_img is not None:
        ph = 60
        pw = int(plate_img.shape[1] * (ph/plate_img.shape[0]))
        plate_small = cv2.resize(plate_img, (pw, ph))
        try:
            img[ y1:y1+ph, x2+5:x2+5+pw ] = plate_small
        except:
            pass

def main(image_path="test1.jpg", out_vis="out_vis1.jpg"):
    os.makedirs("models", exist_ok=True)
    det = VehicleDetector(model_path="models/yolov8n.pt")
    wp = WPODProcessor(model_path="models/wpod-net.h5", plate_size=(240,64))
    ocr = OCRReader(langs=['en'])

    img = cv2.imread(image_path)
    if img is None:
        print("Cannot open image:", image_path)
        return

    boxes = det.detect(image_path)
    print("Detected vehicles:", len(boxes))
    results = []
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box
        # clamp
        x1,x2 = max(0,x1), min(img.shape[1]-1,x2)
        y1,y2 = max(0,y1), min(img.shape[0]-1,y2)
        vehicle = img[y1:y2, x1:x2].copy()
        if vehicle.size == 0:
            continue
        plate = wp.extract_plate(vehicle)
        text = ocr.read_plate(plate) if plate is not None else ""
        print(f"[Vehicle {i+1}] OCR: {text}")
        results.append({'box':box, 'text':text})
        draw_boxes_and_plate(img, box, plate, text, i+1)

    cv2.imwrite(out_vis, img)
    print("Wrote visualization to", out_vis)
    # also show
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", default="test.jpg")
    p.add_argument("--out", "-o", default="out_vis.jpg")
    args = p.parse_args()
    main(args.image, args.out)
