import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd

from src.config import Config
from src.pretrain_dataset import SingleFrameDataset
from src.models.pretrain_cnn import PretrainCNN


def build_pretrain_loaders(batch_size=None, val_ratio=0.1):
    """
    Tạo train/val loader riêng cho pre-train ảnh đơn.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Ưu tiên đường dẫn hiện có trong config, fallback nếu thư mục ảnh khác tên.
    images_root = Config.IMAGES_ROOT
    if not os.path.exists(images_root):
        alt_root = os.path.join(os.path.dirname(Config.TRAIN_CSV), "images_resized")
        if os.path.exists(alt_root):
            images_root = alt_root

    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = train_test_split(full_df, test_size=val_ratio, random_state=42)

    train_ds = SingleFrameDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=images_root,
        telemetry_root=Config.TELEMETRY_ROOT,
        transform=transform,
        timestamp_mode="mid",
    )
    train_ds.data = train_df.reset_index(drop=True)

    val_ds = SingleFrameDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=images_root,
        telemetry_root=Config.TELEMETRY_ROOT,
        transform=transform,
        timestamp_mode="mid",
    )
    val_ds.data = val_df.reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


import matplotlib.pyplot as plt
import os

def plot_metrics(history: dict, save_dir: str = "."):
    """
    Vẽ và lưu biểu đồ MSE + MAE sau khi train xong.

    history = {
        'train_mse': [...], 'val_mse': [...],
        'train_mae': [...], 'val_mae': [...]
    }
    """
    epochs = range(1, len(history['train_mse']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Metrics", fontsize=14, fontweight='bold')

    # ── MSE ──
    axes[0].plot(epochs, history['train_mse'], label='Train MSE', color='#3266ad', linewidth=2)
    axes[0].plot(epochs, history['val_mse'],   label='Val MSE',   color='#c95a3a', linewidth=2)
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── MAE ──
    axes[1].plot(epochs, history['train_mae'], label='Train MAE', color='#1d9e75', linewidth=2)
    axes[1].plot(epochs, history['val_mae'],   label='Val MAE',   color='#ba7517', linewidth=2)
    axes[1].set_title("MAE Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved chart → {save_path}")


def run_pretrain(train_loader, val_loader, epochs=10, lr=1e-4, device=None, save_path="cnn_pretrained.pth"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PretrainCNN().to(device)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    # ── Lưu lịch sử để vẽ biểu đồ ──
    history = {'train_mse': [], 'val_mse': [], 'train_mae': [], 'val_mae': []}

    for epoch in range(epochs):
        model.train()
        train_mse, train_mae = 0.0, 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion_mse(preds, targets)
            loss.backward()
            optimizer.step()
            train_mse += loss.item()
            train_mae += criterion_mae(preds, targets).item()

        avg_train_mse = train_mse / max(1, len(train_loader))
        avg_train_mae = train_mae / max(1, len(train_loader))

        model.eval()
        val_mse, val_mae = 0.0, 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                val_mse += criterion_mse(preds, targets).item()
                val_mae += criterion_mae(preds, targets).item()

        avg_val_mse = val_mse / max(1, len(val_loader))
        avg_val_mae = val_mae / max(1, len(val_loader))

        # ── Ghi vào history ──
        history['train_mse'].append(avg_train_mse)
        history['val_mse'].append(avg_val_mse)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"| Train MSE: {avg_train_mse:.6f}  Train MAE: {avg_train_mae:.6f} "
            f"| Val MSE: {avg_val_mse:.6f}  Val MAE: {avg_val_mae:.6f}",
            flush=True
        )

        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (Val MSE={best_val_loss:.6f})")

    # ── Vẽ biểu đồ sau khi train xong ──
    plot_metrics(history, save_dir=os.path.dirname(save_path))

if __name__ == "__main__":
    train_loader, val_loader = build_pretrain_loaders()
    run_pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=Config.DEVICE,
        save_path = os.path.join(os.path.dirname(__file__), "saved_models", "cnn_pretrained.pth")
    )
    