#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入本地模块
from model import create_model
from data_loader import load_data

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 训练一个周期
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # 更新进度条
        pbar.set_postfix({"loss": loss.item(), "acc": torch.sum(preds == labels.data).item() / inputs.size(0)})
    
    # 计算整个周期的损失和准确率
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc.item()

# 在验证集上评估模型
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # 禁用梯度计算
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    # 计算整个数据集的损失和准确率
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    
    return epoch_loss, epoch_acc.item()

# 绘制训练过程的损失和准确率曲线
def plot_training_history(train_loss, val_loss, train_acc, val_acc, save_path):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train a lightweight model for image classification')
    parser.add_argument('--data_dir', default='./data/images', help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--model', type=str, default='mobilenetv3_small', 
                        choices=['mobilenetv2', 'shufflenetv2', 'squeezenet', 'mobilenetv3_small', 
                                 'tinyconvnet', 'micronet'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory for models and plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # 添加特定模型的参数
    parser.add_argument('--shufflenet_size', type=str, default='1.0x', 
                        choices=['0.5x', '1.0x', '1.5x', '2.0x'],
                        help='Size variant for ShuffleNetV2')
    parser.add_argument('--squeezenet_version', type=str, default='1_1', 
                        choices=['1_0', '1_1'],
                        help='Version for SqueezeNet')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"确保输出目录存在: {args.output_dir}")
    
    # 创建模型保存子目录
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"确保模型保存目录存在: {models_dir}")
    
    # 创建可视化结果子目录
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"确保可视化结果目录存在: {plots_dir}")
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_names = load_data(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"类别: {class_names}")
    print(f"训练集大小: {len(train_loader.dataset)} 图像")
    print(f"验证集大小: {len(val_loader.dataset)} 图像")  
    print(f"测试集大小: {len(test_loader.dataset)} 图像")
    
    # 创建模型
    model_kwargs = {}
    if args.model == 'shufflenetv2':
        model_kwargs['size'] = args.shufflenet_size
    elif args.model == 'squeezenet':
        model_kwargs['version'] = args.squeezenet_version
        
    model = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        **model_kwargs
    )
    model = model.to(device)
    
    model_name = args.model
    if args.model == 'shufflenetv2':
        model_name += f"_{args.shufflenet_size}"
    elif args.model == 'squeezenet':
        model_name += f"_{args.squeezenet_version}"
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_acc': 0.0
    }
    
    # 训练循环
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练一个周期
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))
            print(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
    
    # 打印训练总时间
    time_elapsed = time.time() - start_time
    print(f'\n训练完成，用时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"最佳验证准确率: {history['best_val_acc']:.4f}")
    
    # 绘制训练历史
    plot_training_history(
        history['train_loss'], 
        history['val_loss'], 
        history['train_acc'], 
        history['val_acc'],
        os.path.join(plots_dir, 'training_history.png')
    )
    
    # 加载最佳模型进行测试
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n测试结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    # 保存最终模型（用于后续转换为NCNN格式）
    final_model_path = os.path.join(models_dir, f'{model_name}_model.pth')
    
    # 使用更兼容的方式保存模型
    torch.save({
        'state_dict': model.state_dict(),
        'model_name': args.model,
        'model_kwargs': model_kwargs,
        'num_classes': args.num_classes
    }, final_model_path)
    print(f"模型保存在 {final_model_path}")
    
    # 保存类别名称列表（用于部署时的推理）
    with open(os.path.join(args.output_dir, 'class_names.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # 导出ONNX格式（为后续转换为NCNN做准备）
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = os.path.join(models_dir, f'{model_name}_model.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX模型保存在 {onnx_path}")

if __name__ == "__main__":
    main()