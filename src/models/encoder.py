import torch
import torch.nn as nn
import torchvision.models as models


class MultimodalEncoder(nn.Module):
    """
    Multimodal Encoder với Early Fusion.
    Nối đặc trưng Image (512-d) + Sensor (3-d) = 515-d rồi đưa qua LSTM 2 tầng.
    """

    def __init__(self, hidden_size=1024, sensor_dim=3):
        super(MultimodalEncoder, self).__init__()

        # --- NHÁNH HÌNH ẢNH (CNN Feature Extractor) ---
        resnet = models.resnet18(pretrained=True)
        # Bỏ lớp FC cuối (classification head) của ResNet
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        # Output ResNet18: [*, 512, 1, 1] -> squeeze -> [*, 512]

        # --- EARLY FUSION LSTM ---
        # Input size = ResNet output (512) + Sensor (3) = 515
        fusion_input_dim = 512 + sensor_dim
        self.lstm = nn.LSTM(
            input_size=fusion_input_dim,  # 515
            hidden_size=hidden_size,      # 1024
            num_layers=2,                 # 2 tầng LSTM
            batch_first=True
        )

    def forward(self, images, sensors):
        """
        Args:
            images:  [Batch, 16, 3, 224, 224]
            sensors: [Batch, 16, 3]  (speed, acceleration, course)
        Returns:
            context_vector: [Batch, 1024]
        """
        batch_size, frames, C, H, W = images.shape

        # --- A. TRÍCH XUẤT ĐẶC TRƯNG ẢNH ---
        # Gộp Batch*Frames để đưa qua CNN một lượt
        c_in = images.view(batch_size * frames, C, H, W)  # shape: [B*16, 3, 224, 224]

        with torch.no_grad():
            features = self.cnn(c_in)  # shape: [B*16, 512, 1, 1]

        features = features.view(features.size(0), -1)           # shape: [B*16, 512]
        features = features.view(batch_size, frames, -1)         # shape: [B, 16, 512]

        # --- B. EARLY FUSION: NỐI IMAGE + SENSOR ---
        # sensors: [B, 16, 3]
        fused = torch.cat((features, sensors), dim=2)            # shape: [B, 16, 515]

        # --- C. LSTM 2 TẦNG ---
        # lstm_out: [B, 16, 1024] (output tại mọi timestep)
        # h_n:      [2, B, 1024]  (hidden state cuối của mỗi tầng)
        lstm_out, (h_n, c_n) = self.lstm(fused)

        # Lấy hidden state cuối cùng của TẦNG THỨ 2 (index -1)
        context_vector = h_n[-1]                                 # shape: [B, 1024]

        return context_vector