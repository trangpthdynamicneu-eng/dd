# wpod_net.py
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

class WPODProcessor:
    def __init__(self, model_path="models/wpod-net.h5", plate_size=(240,64)):
        """
        model_path: pretrained wpod-net .h5 (if present we use its presence output if possible)
        plate_size: output standardized plate size (w,h)
        """
        self.plate_w, self.plate_h = plate_size
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, compile=False)
                print("[WPOD] Loaded model:", model_path)
            except Exception as e:
                print(f"[WPOD] Failed to load model: {e}. WPOD model disabled; will use heuristic.")
        else:
            print("[WPOD] model file not found. WPOD model disabled; will use heuristic detection only.")

    def _run_model(self, crop_rgb):
        """
        Run model if available. Return raw output as numpy array, or None.
        We will attempt to obtain a presence heatmap from outputs.
        """
        if self.model is None:
            return None
        # model likely expects float normalized image and specific input shape.
        inp = cv2.resize(crop_rgb, (208, 208))   # WPOD original used 208x208 (approx)
        x = inp.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        try:
            y = self.model.predict(x)
            # y might be a list or single array. Normalize to array.
            if isinstance(y, list):
                y = y[0]
            return y
        except Exception as e:
            print("[WPOD] Model predict failed:", e)
            return None

    def _heatmap_from_model_output(self, raw_out, target_shape):
        """
        Try to derive a presence heatmap from raw_out.
        Many WPOD implementations output a map where channel 0 or channel 0..1 indicate presence.
        We'll try typical possibilities and resize to target_shape.
        """
        if raw_out is None:
            return None
        # raw_out shape could be (1, H, W, C) or (1, C, H, W)
        arr = np.array(raw_out)
        if arr.ndim == 4:
            # (1,H,W,C) or (1,C,H,W)
            if arr.shape[1] < 10:  # maybe (1,C,H,W)
                # transpose to (1,H,W,C)
                arr = np.transpose(arr, (0,2,3,1))
            arr = arr[0]  # H,W,C
            # heuristic: choose channel with max variance or the first channel
            ch_idx = 0
            # if there are >1 channels, maybe first channel is presence; else take mean
            if arr.shape[2] >= 1:
                heat = arr[:,:,ch_idx]
            else:
                heat = np.mean(arr, axis=2)
            heat = cv2.resize(heat, (target_shape[1], target_shape[0]))
            # normalize 0..1
            heat = heat - heat.min()
            if heat.max() > 0:
                heat = heat / heat.max()
            return heat
        elif arr.ndim == 3:
            # (H,W,C) or (C,H,W)
            if arr.shape[0] <= 4:  # likely (C,H,W)
                arr = np.transpose(arr, (1,2,0))  # to H,W,C
            # take first channel
            heat = arr[:,:,0]
            heat = cv2.resize(heat, (target_shape[1], target_shape[0]))
            heat = heat - heat.min()
            if heat.max() > 0:
                heat = heat / heat.max()
            return heat
        else:
            return None

    def _heuristic_heatmap(self, crop_gray):
        """
        Simple heuristic when model absent: detect high edge-density regions.
        Returns normalized heatmap same size as crop_gray.
        """
        edges = cv2.Canny(crop_gray, 50, 150)
        # morphological closing + gaussian blur to make blob-like
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
        m = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        m = cv2.GaussianBlur(m.astype(np.float32), (9,9), 0)
        heat = m - m.min()
        if heat.max() > 0:
            heat = heat / heat.max()
        return heat

    def extract_plate(self, vehicle_bgr):
        """
        Input: vehicle_bgr (numpy BGR crop)
        Output: warped plate image (BGR) of size (plate_h, plate_w) or None
        """
        # prepare
        h,w = vehicle_bgr.shape[:2]
        if h < 10 or w < 10:
            return None

        crop_rgb = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2RGB)
        raw_out = self._run_model(crop_rgb)
        heat = self._heatmap_from_model_output(raw_out, target_shape=(h,w))
        if heat is None:
            # fallback heuristic
            crop_gray = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2GRAY)
            heat = self._heuristic_heatmap(crop_gray)

        # threshold heat to get candidate region(s)
        thresh = 0.35
        binm = (heat >= thresh).astype(np.uint8) * 255
        # if no pixel, lower threshold
        if binm.sum() == 0:
            thresh = 0.2
            binm = (heat >= thresh).astype(np.uint8) * 255
        # morphological to connect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,5))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # try largest connected high-value area by centering on global max
            (my,mx) = np.unravel_index(np.argmax(heat), heat.shape)
            # make small box around it
            bw = max(20, w//6)
            bh = max(10, h//10)
            x1 = max(0, mx - bw//2); x2 = min(w, mx + bw//2)
            y1 = max(0, my - bh//2); y2 = min(h, my + bh//2)
            candidate = vehicle_bgr[y1:y2, x1:x2]
            # warp upright by simple resize
            warped = cv2.resize(candidate, (self.plate_w, self.plate_h))
            return warped

        # pick contour with largest area * aspect-likeness (prefer wide rectangles)
        best_cnt = None
        best_score = -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            rect = cv2.minAreaRect(cnt)
            (rw,rh) = rect[1]
            if rw <= 0 or rh <= 0:
                continue
            aspect = max(rw,rh) / (min(rw,rh)+1e-6)
            # prefer aspect ratio typical of plates (around 2..8 depending)
            ar_score = 1.0 - abs(aspect - 4.0) / 10.0
            score = area * max(0.1, ar_score)
            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is None:
            return None

        # get rotated rect and box points
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # 4 points

        # order box points: top-left, top-right, bottom-right, bottom-left
        def order_pts(pts):
            rect = np.zeros((4,2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        src_pts = order_pts(box)
        dst_pts = np.array([
            [0,0],
            [self.plate_w-1, 0],
            [self.plate_w-1, self.plate_h-1],
            [0, self.plate_h-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(vehicle_bgr, M, (self.plate_w, self.plate_h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return warped
