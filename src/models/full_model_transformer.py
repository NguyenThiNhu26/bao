import torch
import torch.nn as nn
from timm import create_model

from src.models.action_head import ActionRegressor   # Giữ nguyên Action Head cũ của bạn


class DrivingRiskModelTransformer(nn.Module):
    """
    Mức 2 - Kiến trúc hiện đại 2025:
    - Swin Transformer (vision backbone)
    - Temporal Transformer Encoder
    - Transformer Decoder (thay LSTM)
    - Vẫn giữ Action Regressor cũ
    """

    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.HIDDEN_SIZE

        # ====================== VISION BACKBONE ======================
        # Swin Tiny pretrained (rất mạnh hơn CNN 5 lớp cũ)
        self.vision_backbone = create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0
        )
        self.visual_proj = nn.Linear(768, self.hidden_size)   # 768 -> 1024

        # ====================== TEMPORAL ENCODER ======================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ====================== ACTION REGRESSOR (GIỮ NGUYÊN) ======================
        self.action_head = ActionRegressor(
            hidden_size=self.hidden_size,
            future_steps=config.FUTURE_STEPS
        )

        # ====================== CAPTION DECODER (Transformer) ======================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.caption_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.word_embed = nn.Embedding(vocab_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, vocab_size)

        # Positional embedding cho các frame
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, config.MAX_FRAMES, self.hidden_size))

    def forward(self, images, sensors, captions):
        """
        images:   [B, T, 3, H, W]
        sensors:  [B, T, 3]   (hiện tại chưa dùng mạnh, có thể mở rộng sau)
        captions: [B, SeqLen]
        """
        B, T, C, H, W = images.shape

        # 1. Trích xuất đặc trưng hình ảnh từng frame
        x = images.view(B * T, C, H, W)           # [B*T, 3, H, W]
        x = self.vision_backbone(x)               # [B*T, 768]
        x = self.visual_proj(x)                   # [B*T, hidden]
        x = x.view(B, T, -1)                      # [B, T, hidden]

        # 2. Thêm positional embedding + Temporal Transformer
        x = x + self.frame_pos_embed[:, :T, :]
        context_seq = self.temporal_encoder(x)    # [B, T, hidden]
        context = context_seq.mean(dim=1)         # [B, hidden] - global context

        # 3. Action Regressor
        future_flat = self.action_head(context)
        future_pred = self.action_head.reshape_prediction(future_flat)

        # 4. Caption Decoder (Teacher-forcing)
        tgt = self.word_embed(captions)           # [B, SeqLen, hidden]
        memory = context.unsqueeze(1)             # [B, 1, hidden] dùng làm memory
        decoder_output = self.caption_decoder(tgt, memory)

        vocab_outputs = self.fc_out(decoder_output)   # [B, SeqLen, vocab_size]

        return vocab_outputs, future_pred