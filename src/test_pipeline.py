# test_pipeline.py
import argparse
import cv2
import matplotlib.pyplot as plt
from pipeline import main

def run_test(image_path, out_path):
    """
    Chạy pipeline và hiển thị kết quả
    """
    print(f"[TEST] Input image: {image_path}")
    print(f"[TEST] Output will be saved as: {out_path}")

    # Gọi pipeline chính (hàm main trong pipeline.py)
    main(image_path, out_path)

    # Đọc ảnh kết quả để hiển thị
    result = cv2.imread(out_path)
    if result is None:
        print("[TEST] Không thể đọc ảnh output.")
        return

    # Hiển thị bằng matplotlib (thay cv2.imshow để tránh lỗi GUI)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Pipeline Result")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Đường dẫn ảnh input")
    parser.add_argument("--out", "-o", default="test_result.jpg", help="Đường dẫn ảnh output")
    args = parser.parse_args()

    run_test(args.image, args.out)
