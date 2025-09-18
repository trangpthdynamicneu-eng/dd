# wpod_arch.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Input

def conv_block(x, filters, kernel=3, strides=1, bn=True):
    x = Conv2D(filters, kernel, strides=strides, padding="same")(x)
    if bn:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def build_wpod(input_shape=(208, 208, 3)):
    """
    Build WPOD-Net architecture
    Input: ảnh phương tiện (208x208x3)
    Output: feature map dùng để dự đoán bounding box biển số
    """
    x_in = Input(shape=input_shape)

    # Conv Block 1
    x = conv_block(x_in, 16)
    x = conv_block(x, 16)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 4
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 5
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    # Lớp đầu ra
    # WPOD-Net gốc xuất ra 8 giá trị (tọa độ 4 góc biển số sau warp)
    out = Conv2D(8, (1, 1), activation=None, padding="same")(x)

    model = Model(x_in, out, name="WPOD-NET")
    return model
