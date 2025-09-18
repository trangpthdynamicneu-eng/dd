from detector import VehicleDetector
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Ảnh test (bạn thay bằng ảnh của mình, ví dụ 'sample.jpg')
    image_path = "images\img.png"

    # Khởi tạo detector
    detector = VehicleDetector(model_path="models/yolov8n.pt", device="cpu")

    # Gọi hàm detect
    boxes = detector.detect(image_path, conf=0.25)

    print("Số phương tiện phát hiện:", len(boxes))
    for i, box in enumerate(boxes, 1):
        print(f"Xe {i}: {box}")

    # Vẽ bounding box để xem trực quan
    image = cv2.imread(image_path)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Lưu ảnh ra file
    cv2.imwrite("output.jpg", image)
    print("Ảnh kết quả đã lưu tại: output.jpg")

    # # Hiển thị bằng matplotlib (không bị lỗi GUI)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()
