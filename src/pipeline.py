# pipeline.py
import cv2
import os
from detector import VehicleDetector   # Module phát hiện phương tiện (sử dụng YOLOv8)
from wpod_net import WPODProcessor     # Module cắt & hiệu chỉnh biển số (WPOD-Net)
from ocr_reader import OCRReader       # Module OCR đọc ký tự (EasyOCR)


def draw_boxes_and_plate(img, box, plate_img, text, idx):
    """
    Hàm vẽ bounding box quanh phương tiện, chèn biển số và text OCR vào ảnh gốc.
    :param img: ảnh gốc
    :param box: tọa độ khung bao (x1, y1, x2, y2)
    :param plate_img: ảnh biển số cắt ra
    :param text: kết quả OCR
    :param idx: số thứ tự phương tiện
    """
    x1, y1, x2, y2 = box
    # Vẽ khung xanh quanh phương tiện
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Ghi text OCR + index phía trên khung
    cv2.putText(img, f"{idx}:{text}", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị ảnh biển số nhỏ cạnh phương tiện
    if plate_img is not None:
        ph = 60  # chiều cao hiển thị
        pw = int(plate_img.shape[1] * (ph / plate_img.shape[0]))  # giữ tỉ lệ
        plate_small = cv2.resize(plate_img, (pw, ph))
        try:
            img[y1:y1 + ph, x2 + 5:x2 + 5 + pw] = plate_small
        except:
            # Trường hợp biển số vượt ra ngoài vùng ảnh thì bỏ qua
            pass


def main(image_path="test_vehicle.jpg", out_vis="out_vis.jpg"):
    """
    Hàm chính: chạy pipeline nhận diện biển số
    :param image_path: đường dẫn ảnh đầu vào
    :param out_vis: ảnh kết quả sau khi vẽ bounding box & biển số
    """
    # Đảm bảo thư mục models tồn tại
    os.makedirs("models", exist_ok=True)

    # Khởi tạo các module chính
    det = VehicleDetector(model_path="models/yolov8n.pt")           # Phát hiện xe
    wp = WPODProcessor(model_path="models/wpod-net_update.json",
                       plate_size=(240, 64))
    ocr = OCRReader(langs=['en', 'vi'])                                   # Đọc ký tự biển số

    # Đọc ảnh đầu vào
    img = cv2.imread(image_path)
    if img is None:
        print("Không mở được ảnh:", image_path)
        return

    # B1: Phát hiện phương tiện bằng YOLOv8
    boxes = det.detect(image_path)
    print("Số lượng phương tiện phát hiện:", len(boxes))

    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Đảm bảo tọa độ không vượt ngoài ảnh
        x1, x2 = max(0, x1), min(img.shape[1] - 1, x2)
        y1, y2 = max(0, y1), min(img.shape[0] - 1, y2)

        # Cắt vùng ảnh chứa phương tiện
        vehicle = img[y1:y2, x1:x2].copy()
        if vehicle.size == 0:
            continue

        # B2: Trích xuất biển số từ phương tiện
        plate = wp.extract_plate(vehicle)
        if plate is not None:
            cv2.imwrite(f"debug_plate_{i + 1}.jpg", plate)
            print(f"[DEBUG] Saved cropped plate for vehicle {i + 1}")
        else:
            print(f"[DEBUG] Không cắt được biển số cho vehicle {i + 1}")

        # B3: OCR đọc biển số
        text = ocr.read_plate(plate) if plate is not None else ""
        print(f"[Vehicle {i+1}] OCR: {text}")

        # Lưu kết quả
        results.append({'box': box, 'text': text})

        # Vẽ bounding box, text OCR và hiển thị ảnh biển số
        draw_boxes_and_plate(img, box, plate, text, i + 1)

    # Ghi kết quả ra file ảnh
    cv2.imwrite(out_vis, img)
    print("Đã lưu kết quả trực quan vào:", out_vis)

    # # Hiển thị ảnh kết quả
    # cv2.imwrite(out_vis, img)
    # print("Đã lưu kết quả vào:", out_vis)

    # Nếu muốn hiển thị ngay bằng matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# Chạy trực tiếp bằng command line
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", default="test_vehicle.jpg")     # đường dẫn ảnh input
    p.add_argument("--out", "-o", default="out_vis.jpg")    # đường dẫn ảnh output
    args = p.parse_args()
    main(args.image, args.out)
