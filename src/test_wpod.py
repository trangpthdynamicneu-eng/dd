import cv2
from wpod_net import WPODProcessor

def test_wpod(image_path="src/test_vehicle.jpg"):
    # Khởi tạo bộ xử lý WPOD
    wp = WPODProcessor(model_path="models/wpod-net_update.h5", plate_size=(240, 64))

    # Đọc ảnh xe
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không thể mở ảnh:", image_path)
        return

    # Trích xuất biển số
    plate = wp.extract_plate(img)
    if plate is None:
        print("⚠️ Không tìm thấy biển số trong ảnh:", image_path)
        return

    # Lưu ảnh xe gốc và biển số
    cv2.imwrite("out_vehicle.jpg", img)
    cv2.imwrite("out_plate.jpg", plate)

    print("✅ Đã lưu ảnh kết quả:")
    print(" - out_vehicle.jpg (ảnh xe gốc)")
    print(" - out_plate.jpg (ảnh biển số)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", default="src/test_vehicle.jpg", help="Đường dẫn ảnh xe để test WPOD")
    args = p.parse_args()
    test_wpod(args.image)
