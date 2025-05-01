#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

def create_directory_structure(base_dir='.'):
    """创建项目工作空间需要的目录结构"""
    # 要创建的目录
    directories = [
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'outputs', 'models'),
        os.path.join(base_dir, 'outputs', 'plots'),
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, 'data', 'images'),
        os.path.join(base_dir, 'data', 'images', 'train'),
        os.path.join(base_dir, 'data', 'images', 'val'),
        os.path.join(base_dir, 'data', 'images', 'test'),
        os.path.join(base_dir, 'deploy'),
        os.path.join(base_dir, 'deploy', 'models'),
    ]
    
    # 创建目录
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    print("目录结构创建完成！")

def main():
    parser = argparse.ArgumentParser(description='初始化项目工作空间')
    parser.add_argument('--base-dir', default='.', help='项目根目录')
    args = parser.parse_args()
    
    create_directory_structure(args.base_dir)
    
    print("""
项目工作空间已初始化，目录结构如下：
.
├── outputs/            # 模型输出和结果
│   ├── models/         # 保存的模型
│   └── plots/          # 训练可视化图表
├── data/               # 数据目录
│   └── images/         # 图像数据
│       ├── train/      # 训练集
│       ├── val/        # 验证集
│       └── test/       # 测试集
└── deploy/             # 部署相关代码
    └── models/         # 存放转换后的NCNN模型文件
    
你可以开始使用项目：
1. 下载数据集: python scripts/download_cifar10.py
2. 训练模型: python train/train.py --model mobilenetv3_small
3. 转换模型: python scripts/convert_to_ncnn.py --model ./outputs/models/mobilenetv3_small_model.pth
    """)

if __name__ == "__main__":
    main()
