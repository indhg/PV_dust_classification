"""
主程序入口
- 加载数据 -> 构建模型 -> 训练 -> 评估 -> 保存最佳模型
"""

import os
import torch
from dataset import get_dataloaders
from model import build_model
from train import train
from evaluate import evaluate

def main():
    # 基本配置
    data_dir = './data'
    batch_size = 32
    epochs = 20
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    print(f"训练样本: {len(train_loader.dataset)}, "
          f"验证样本: {len(val_loader.dataset)}, "
          f"测试样本: {len(test_loader.dataset)}")

    # 2. 构建模型
    print("构建模型...")
    model = build_model(num_classes=3)

    # 3. 训练模型
    print("开始训练...")
    history = train(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    # 4. 评估模型
    print("在测试集上评估模型...")
    metrics = evaluate(model, test_loader, device=device)

    # 5. 保存模型权重
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")

if __name__ == '__main__':
    main()