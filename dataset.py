"""
数据集加载模块
- 从 data/train 和 data/val 加载图像
- 手动指定 class_to_idx 保证标签对齐
- 训练集使用数据增强，验证集仅缩放和归一化
- 从训练集随机划分 15% 作为测试集（固定种子）
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir='./data', batch_size=32):
    """
    返回 train_loader, val_loader, test_loader
    参数:
        data_dir: 数据集根目录 (包含 train/ 和 val/ 子文件夹)
        batch_size: 批量大小
    返回:
        三个 DataLoader
    """
    # ImageNet 的均值与标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 手动指定类别到索引的映射，确保 '0_assless'->0, '1_little_ashes'->1, '2_all_ashes'->2
    class_to_idx = {'0_assless': 0, '1_little_ashes': 1, '2_all_ashes': 2}

    # ----- 训练集增强预处理 -----
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                # 统一尺寸
        transforms.RandomHorizontalFlip(p=0.5),       # 随机水平翻转
        transforms.ColorJitter(brightness=0.2,        # 颜色抖动
                               contrast=0.2,
                               saturation=0.2),
        transforms.ToTensor(),                        # 转为 Tensor [0,1]
        transforms.Normalize(mean=mean, std=std)      # ImageNet 归一化
    ])

    # ----- 验证集 / 测试集预处理（无增强） -----
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 加载原始训练集（使用增强 transform）
    train_dataset = datasets.ImageFolder(
        root=f'{data_dir}/train',
        transform=train_transform,
    )

    # 加载验证集
    val_dataset = datasets.ImageFolder(
        root=f'{data_dir}/val',
        transform=eval_transform,
    )

    # ----- 从训练集中分出 15% 作为测试集 -----
    total_len = len(train_dataset)
    test_len = int(total_len * 0.15)
    train_len = total_len - test_len

    # 固定种子保证划分结果可复现
    generator = torch.Generator().manual_seed(42)
    train_subset, test_subset = random_split(
        train_dataset,
        [train_len, test_len],
        generator=generator
    )

    # 获取测试集对应的样本索引
    test_indices = test_subset.indices

    # 测试集应使用无增强的 eval_transform，且不能影响原训练集
    # 因此重新加载一份不带增强的训练数据，取其子集作为测试集
    test_full_dataset = datasets.ImageFolder(
        root=f'{data_dir}/train',
        transform=eval_transform,
    )
    test_dataset = Subset(test_full_dataset, test_indices)

    # ----- 构建 DataLoader -----
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader