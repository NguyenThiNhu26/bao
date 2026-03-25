import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.dataset import DrivingRiskDataset

# ====================== CHỌN MODEL Ở ĐÂY ======================
USE_TRANSFORMER_MODEL = True   # ← Đổi thành False nếu muốn dùng model cũ

if USE_TRANSFORMER_MODEL:
    from src.models.full_model_transformer import DrivingRiskModelTransformer as DrivingRiskModel
    print("🚀 Đang dùng Mức 2: Swin + Transformer (file mới)")
else:
    from src.models.full_model import DrivingRiskModel
    print("📌 Đang dùng model cũ (LSTM)")

def train():
    device = Config.DEVICE
    print(f"Đang sử dụng thiết bị: {device}")
    
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    log_file = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), "training_log.csv")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Motion_Loss_Val,Caption_Loss_Val,LR\n")

    # ====================== DATA ======================
    print("Đang tải và chia dữ liệu...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_df = pd.read_csv(Config.TRAIN_CSV)
    train_df, temp_df = train_test_split(full_df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    test_df.to_csv(os.path.join(os.path.dirname(Config.TRAIN_CSV), "test_data.csv"), index=False)

    train_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS,
        df=train_df
    )
    val_dataset = DrivingRiskDataset(..., df=val_df)   # giống như trên

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ====================== MODEL ======================
    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)

    # Chỉ load pretrained CNN khi dùng model cũ
    if not USE_TRANSFORMER_MODEL:
        pretrain_path = os.path.join(os.path.dirname(__file__), "saved_models", "cnn_pretrained.pth")
        if os.path.exists(pretrain_path):
            model.encoder.load_pretrained_cnn(pretrain_path)
            print(f"✅ Loaded pretrained CNN: {pretrain_path}")

    # ====================== LOSS & OPTIMIZER ======================
    criterion_caption = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_motion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

    # ====================== TRAINING LOOP (giống Mức 1) ======================
    # ... (phần training loop tôi đã tối ưu ở tin nhắn trước, bạn copy phần còn lại từ tin nhắn Mức 1)
    # Tôi rút gọn ở đây để tiết kiệm chỗ, bạn dán phần loop train/val giống hệt tin nhắn trước.

    # (Bạn có thể copy phần TRAINING LOOP từ tin nhắn Mức 1 của tôi)

if __name__ == "__main__":
    train()