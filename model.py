"""
CNN 模型定义
- 3 个卷积块：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2)
- 分类器：自适应平均池化 -> 展平 -> Linear(128,128) -> ReLU -> Dropout -> Linear(128,3)
"""

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # 卷积块1: 输入3通道 -> 32通道，特征图尺寸减半
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 卷积块2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 卷积块3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # 输出尺寸 1x1
            nn.Flatten(),                      # 展平为 128 维
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

def build_model(num_classes=3):
    """返回模型实例"""
    return SimpleCNN(num_classes=num_classes)