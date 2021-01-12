import torch
import torch.nn as nn

class VGGExtracter(nn.module):
    def __init__(self):
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
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
class W2VExtracter(nn.module):
    def __init__(self, hidden_size=512, dropout_rate=0.0):
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, 10, stride=5, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=2, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=2, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=2, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=2, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 2, stride=2, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 2, stride=2, bias=False),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
        )
        self.post_extract_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.post_extract_proj(x)
        return x