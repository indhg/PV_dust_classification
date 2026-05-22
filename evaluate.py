"""
评估模块
- 使用 sklearn 计算准确率、精确率、召回率、F1 分数
- 绘制并保存混淆矩阵图、ROC 曲线图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，解决图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块
import os
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)


def plot_confusion_matrix(cm, class_names, save_dir='checkpoints'):
    """绘制并保存混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在每个格子里标注数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵图已保存至 {save_path}")


def plot_roc_curves(all_labels, all_probs, class_names, save_dir='checkpoints'):
    """绘制并保存多分类 ROC 曲线（one-vs-rest）"""
    os.makedirs(save_dir, exist_ok=True)
    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        # 将当前类视为正类，其余为负类
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    # 画对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC 曲线 (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)

    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC 曲线图已保存至 {save_path}")


def evaluate(model, test_loader, device='cuda'):
    """
    评估模型在测试集上的性能
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    返回:
        字典 {'accuracy':..., 'precision':..., 'recall':..., 'f1_score':...}
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []   # 用于画 ROC 曲线

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)   # 转为概率
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 类别名称
    class_names = ['无灰', '轻度沾灰', '重度沾灰']

    # ---- 计算指标 ----
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # ---- 绘制混淆矩阵 ----
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    # ---- 绘制 ROC 曲线 ----
    plot_roc_curves(all_labels, all_probs, class_names)

    # ---- 打印结果 ----
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print("========== 测试集评估结果 ==========")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f} (macro)")
    print(f"召回率 (Recall):    {recall:.4f} (macro)")
    print(f"F1 分数 (F1-Score): {f1:.4f} (macro)")
    print("====================================")

    return metrics