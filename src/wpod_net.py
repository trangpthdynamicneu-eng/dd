# wpod_net.py
import cv2
import numpy as np
from os.path import splitext
from keras.models import model_from_json


def load_wpod_model(path_json):
    """
    Load WPOD-Net từ file json + h5
    """
    base = splitext(path_json)[0]
    with open(f"{base}.json", "r") as f:
        model_json = f.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights(f"{base}.h5")
    print(f"[WPOD] Loaded model from {base}.json / {base}.h5")
    return model


def im2single(I):
    """
    Chuẩn hóa ảnh về [0,1]
    """
    return I.astype("float32") / 255.0


def order_points(pts):
    """
    Sắp xếp 4 điểm: TL, TR, BR, BL
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


def warp_plate(I, pts, out_size=(240, 80)):
    """
    Warp biển số về ảnh thẳng chuẩn kích thước out_size (w,h)
    """
    dst_pts = np.array([
        [0, 0],
        [out_size[0] - 1, 0],
        [out_size[0] - 1, out_size[1] - 1],
        [0, out_size[1] - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(order_points(pts), dst_pts)
    warped = cv2.warpPerspective(I, M, out_size,
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped


def detect_plate(wpod_model, vehicle_bgr, Dmax=608, Dmin=288, lp_threshold=0.5):
    """
    Detect và cắt biển số từ ảnh phương tiện (BGR).
    Trả về list ảnh biển số đã warp.
    """
    # Resize input ảnh xe
    Ivehicle = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2RGB)
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    # Chuẩn hóa
    Iresized = cv2.resize(Ivehicle, (bound_dim, bound_dim))
    T = im2single(Iresized)
    T = np.expand_dims(T, axis=0)

    # Predict bằng WPOD
    Yr = wpod_model.predict(T)
    Yr = np.squeeze(Yr)

    # ---- reconstruct (cách làm gọn, tương tự đoạn 1) ----
    # Lấy heatmap
    Probs = Yr[..., 0]
    xx, yy = np.where(Probs > lp_threshold)
    if len(xx) == 0:
        return []

    # Chọn điểm có xác suất cao nhất
    idx = np.argmax(Probs[xx, yy])
    x, y = xx[idx], yy[idx]

    # Box đơn giản quanh cell có max prob
    h, w = Iresized.shape[:2]
    bw, bh = w // 6, h // 10
    x1, y1 = max(0, y * 4 - bw // 2), max(0, x * 4 - bh // 2)
    x2, y2 = min(w, y * 4 + bw // 2), min(h, x * 4 + bh // 2)

    # Crop candidate plate
    candidate = Iresized[y1:y2, x1:x2]

    # Giả sử box = 4 đỉnh chữ nhật (bạn có thể thay bằng pts từ affine trong reconstruct)
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

    # Warp biển số
    warped = warp_plate(Iresized, pts, out_size=(240, 80))
    return [cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)]

class WPODProcessor:
    def __init__(self, model_path, plate_size=(240, 80)):
        """
        Khởi tạo WPOD Processor
        :param model_path: đường dẫn file json hoặc h5
        :param plate_size: kích thước output plate (w,h)
        """
        self.model = load_wpod_model(model_path)   # tự động load .json + .h5
        self.plate_size = plate_size

    def extract_plate(self, vehicle_img):
        """
        Trích xuất biển số từ ảnh phương tiện (BGR)
        :param vehicle_img: ảnh input BGR
        :return: ảnh biển số (BGR) hoặc None
        """
        plates = detect_plate(self.model, vehicle_img,
                              Dmax=608, Dmin=288, lp_threshold=0.5)
        if len(plates) > 0:
            # Resize về đúng plate_size
            plate = cv2.resize(plates[0], self.plate_size)
            return plate
        return None