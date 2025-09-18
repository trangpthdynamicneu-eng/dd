import json
from tensorflow.keras.models import model_from_json

# Đường dẫn
json_path = "models/wpod-net_update.json"
h5_path   = "models/wpod-net_update.h5"

print("[INFO] Load WPOD architecture from JSON ...")
with open(json_path, "r") as f:
    model_json = f.read()

# Build model từ JSON
wpod_model = model_from_json(model_json)

print("[INFO] Load weights from H5 ...")
wpod_model.load_weights(h5_path)

print("✅ WPOD model loaded successfully!")

# (optional) Lưu lại thành 1 file keras chuẩn
wpod_model.save("models/wpod-net-converted.h5")
print("✅ Saved converted model to models/wpod-net-converted.h5")
