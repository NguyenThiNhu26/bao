import torch
import os

basedir = os.path.dirname(os.path.dirname(__file__))

class Config:
    # ==================== ĐƯỜNG DẪN ====================
    TRAIN_CSV = os.path.join(basedir, 'data', 'processed_train.csv')
    IMAGES_ROOT = os.path.join(basedir, 'data', 'images')
    TELEMETRY_ROOT = os.path.join(basedir, 'data', 'telemetry')

    # ==================== CẤU HÌNH MODEL ====================
    # Kích thước ảnh đầu vào 
    # → Model cũ (CNN): (90, 160)
    # → Model mới (Swin Transformer): PHẢI DÙNG (224, 224)
    IMAGE_SIZE = (224, 224)          # ← ĐÃ ĐỔI CHO MỨC 2

    # Số frame model sẽ nhìn (Start -> Mid)
    MAX_FRAMES = 16

    # Kích thước vector nhúng & hidden
    EMBED_SIZE = 256
    HIDDEN_SIZE = 1024

    # Sensor
    SENSOR_DIM = 3

    # Action Regressor
    FUTURE_STEPS = 5

    # ==================== HUẤN LUYỆN ====================
    BATCH_SIZE = 32                  # ← Giảm từ 50 xuống 32 (Swin nặng hơn)
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4             # ← Giảm LR cho Transformer ổn định hơn

    # Thiết bị
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Save
    MODEL_SAVE_PATH = 'saved_models/best_model_transformer.pth'   # ← Đổi tên để phân biệt