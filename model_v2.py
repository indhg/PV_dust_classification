import torch
import torch.nn as nn


class DeepCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DeepCNN, self).__init__()
        # 卷积块1: 输入3通道 -> 32通道
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
        # 卷积块4: 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 卷积块5: 256 -> 512
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出尺寸 1x1
            nn.Flatten(),  # 展平为 512 维
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=3):
    """返回模型实例"""
    return DeepCNN(num_classes=num_classes)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = build_model(num_classes=3)

    # 创建测试输入 (batch_size=1, 3通道, 224x224)
    test_input = torch.randn(1, 3, 224, 224)

    # 前向传播
    output = model(test_input)

    # 打印结果
    print("模型结构：")
    print(model)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output}")

    # 验证输出形状
    assert output.shape == (1, 3), f"输出形状应为(1, 3)，但得到{output.shape}"
    print("✅ 测试通过：输出形状正确！")