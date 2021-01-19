import torch
import torch.nn as nn

class VGGExtractor(nn.Module):
    def __init__(self, feature_size=128, hidden_size=512, proj_dropout_rate=0.1):
        super(VGGExtracter).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.post_extract_proj = nn.Linear(feature_size, hidden_size)
        self.dropout = nn.Dropout(p=proj_dropout_rate)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.post_extract_proj(x)
        x = self.dropout(x)
        return x
        
class W2VExtractor(nn.Module):
    def __init__(self, feature_size=512, hidden_size=512, conv_dropout_rate=0.0, proj_dropout_rate=0.1):
        super(W2VExtracter).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, feature_size, 10, stride=5, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 2, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 2, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
        )
        self.post_extract_proj = nn.Linear(feature_size, hidden_size)
        self.dropout = nn.Dropout(p=proj_dropout_rate)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.post_extract_proj(x)
        x = self.dropout(x)
        return x