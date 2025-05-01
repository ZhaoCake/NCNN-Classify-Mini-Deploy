#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_transforms():
    """
    创建训练和验证/测试时使用的数据变换
    
    返回:
        train_transform: 训练数据变换
        val_transform: 验证/测试数据变换
    """
    # 训练数据增强和归一化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV2需要224x224的输入
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])  # ImageNet预训练模型的归一化参数
    ])
    
    # 验证/测试数据只需要调整大小和归一化
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_data(data_dir, batch_size=32, num_workers=4):
    """
    加载训练、验证和测试数据集
    
    参数:
        data_dir: 包含train、val和test文件夹的数据目录
        batch_size: 批处理大小
        num_workers: 数据加载使用的线程数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        class_names: 类别名称列表
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # 获取数据变换
    train_transform, val_transform = get_data_transforms()
    
    # 创建数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 获取类别名称
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names

# 用于测试数据加载
if __name__ == "__main__":
    # 假设数据位于当前目录下的data/images文件夹
    data_dir = "./data/images"
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, class_names = load_data(data_dir)
    
    # 打印数据集信息
    print(f"类别: {class_names}")
    print(f"训练集大小: {len(train_loader.dataset)} 图像")
    print(f"验证集大小: {len(val_loader.dataset)} 图像")  
    print(f"测试集大小: {len(test_loader.dataset)} 图像")
    
    # 检查图像数据的形状
    for images, labels in train_loader:
        print(f"批次大小: {images.shape}")
        print(f"标签形状: {labels.shape}")
        break  # 只检查第一个批次