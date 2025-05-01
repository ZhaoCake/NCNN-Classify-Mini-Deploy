#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import torchvision
from torchvision import datasets
from PIL import Image

def download_cifar10(download_dir='./data'):
    """下载CIFAR10数据集并保存到指定目录"""
    # 确保输出目录存在
    os.makedirs(download_dir, exist_ok=True)
    print(f"确保数据输出目录存在: {download_dir}")
    
    print("正在下载CIFAR10数据集...")
    
    # 下载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root=download_dir, train=True,
                                           download=True)
    testset = torchvision.datasets.CIFAR10(root=download_dir, train=False,
                                          download=True)
    
    print(f"CIFAR10数据集下载完成，保存在 {download_dir} 目录")
    return trainset, testset

def convert_to_image_format(dataset, output_dir, dataset_type='train'):
    """将CIFAR10数据集转换为图像文件并按类别整理"""
    # 类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建输出目录
    base_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 为每个类别创建子目录
    for cls in classes:
        os.makedirs(os.path.join(base_output_dir, cls), exist_ok=True)
    
    print(f"正在处理{dataset_type}集...")
    
    # 将数据集转换为图片
    for i, (img, label) in enumerate(zip(dataset.data, dataset.targets)):
        # 转换为PIL图像
        img = Image.fromarray(img)
        
        # 保存图像
        img_path = os.path.join(base_output_dir, classes[label], f"{i}.png")
        img.save(img_path)
        
        # 打印进度
        if (i+1) % 1000 == 0:
            print(f"已处理 {i+1}/{len(dataset.data)} 张图像")
    
    print(f"{dataset_type}集处理完成，共 {len(dataset.data)} 张图像")

def main():
    # 数据路径
    data_dir = './data'
    image_dir = './data/images'
    
    # 下载数据集
    trainset, testset = download_cifar10(data_dir)
    
    # 转换为图片格式
    convert_to_image_format(trainset, image_dir, 'train')
    convert_to_image_format(testset, image_dir, 'test')
    
    # 创建验证集（从训练集中划分20%）
    train_image_dir = os.path.join(image_dir, 'train')
    val_image_dir = os.path.join(image_dir, 'val')
    
    # 创建验证集目录
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    for cls in classes:
        os.makedirs(os.path.join(val_image_dir, cls), exist_ok=True)
        
        # 获取当前类别的所有训练图像
        cls_dir = os.path.join(train_image_dir, cls)
        image_files = os.listdir(cls_dir)
        
        # 计算要移动到验证集的图像数量（20%）
        val_count = int(len(image_files) * 0.2)
        
        # 随机选择图像移动到验证集
        np.random.shuffle(image_files)
        val_images = image_files[:val_count]
        
        # 移动图像到验证集
        for img in val_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(val_image_dir, cls, img)
            shutil.move(src, dst)
        
        print(f"类别 {cls}: 移动了 {val_count} 张图像到验证集")
    
    print("数据集处理完成!")
    print(f"训练集、验证集和测试集分别保存在 {train_image_dir}、{val_image_dir} 和 {os.path.join(image_dir, 'test')} 目录")

if __name__ == "__main__":
    main()