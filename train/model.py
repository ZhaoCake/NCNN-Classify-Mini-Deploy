#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models

def create_mobilenetv2_model(num_classes=10, pretrained=True):
    """
    创建并返回一个MobileNetV2模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的MobileNetV2模型
    """
    # 加载模型，可选是否使用预训练权重
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # 修改最后的分类器层以适应我们的类别数量
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def create_shufflenetv2_model(num_classes=10, pretrained=True, size='1.0x'):
    """
    创建并返回一个ShuffleNetV2模型，比MobileNetV2更轻量
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
        size: 模型大小，可选 '0.5x', '1.0x', '1.5x', '2.0x'
    
    返回:
        配置好的ShuffleNetV2模型
    """
    if size == '0.5x':
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
    elif size == '1.0x':
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif size == '1.5x':
        model = models.shufflenet_v2_x1_5(pretrained=pretrained)
    elif size == '2.0x':
        model = models.shufflenet_v2_x2_0(pretrained=pretrained)
    else:
        raise ValueError("size must be one of '0.5x', '1.0x', '1.5x', '2.0x'")
    
    # 修改分类器以匹配所需的类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def create_squeezenet_model(num_classes=10, pretrained=True, version='1_1'):
    """
    创建并返回一个SqueezeNet模型，非常轻量
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
        version: 模型版本，'1_0'或'1_1'(1_1更快更准确)
    
    返回:
        配置好的SqueezeNet模型
    """
    if version == '1_0':
        model = models.squeezenet1_0(pretrained=pretrained)
    elif version == '1_1':
        model = models.squeezenet1_1(pretrained=pretrained)
    else:
        raise ValueError("version must be '1_0' or '1_1'")
    
    # SqueezeNet的分类器替换方式不同
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    
    return model

def create_mobilenetv3_small_model(num_classes=10, pretrained=True):
    """
    创建并返回一个MobileNetV3-Small模型，比MobileNetV2更轻量
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的MobileNetV3-Small模型
    """
    model = models.mobilenet_v3_small(pretrained=pretrained)
    
    # 修改分类器以匹配所需的类别数
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

class TinyConvNet(nn.Module):
    """
    自定义的超轻量级卷积神经网络，比标准模型小10-20倍
    专为简单图像分类任务设计，适合资源极度受限的环境
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(TinyConvNet, self).__init__()
        
        # 基础通道数量，影响模型大小
        base_channels = 16
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 第二个卷积块
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            # 第三个卷积块
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            
            # 第四个卷积块
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            
            # 使用全局平均池化代替全连接层，可以大大减少参数量
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_channels*8, num_classes)
        )

    def forward(self, x):
        # 应用特征提取
        x = self.features(x)
        # 展平但保持批次维度
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x

class MicroNet(nn.Module):
    """
    比TinyConvNet更轻量的极简网络，专为极小型设备优化
    参数量只有几万个，适合资源极度受限的微控制器
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(MicroNet, self).__init__()
        
        # 极小通道数
        base_channels = 8
        
        # 极简特征提取
        self.features = nn.Sequential(
            # 快速下采样以减少计算量
            nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 简化的分类器
        self.classifier = nn.Linear(base_channels*4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_custom_tiny_model(num_classes=10, model_type='tinyconvnet'):
    """
    创建自定义的超轻量级模型
    
    参数:
        num_classes: 分类类别数量
        model_type: 模型类型，'tinyconvnet'或'micronet'
    
    返回:
        配置好的超轻量级模型
    """
    if model_type == 'tinyconvnet':
        return TinyConvNet(num_classes=num_classes)
    elif model_type == 'micronet':
        return MicroNet(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的自定义模型类型: {model_type}")

def create_model(model_name='mobilenetv2', num_classes=10, pretrained=True, **kwargs):
    """
    根据名称创建并返回指定的模型
    
    参数:
        model_name: 模型名称，支持 'mobilenetv2', 'shufflenetv2', 'squeezenet', 'mobilenetv3_small', 'tinyconvnet', 'micronet'
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
        **kwargs: 其他特定模型的参数
    
    返回:
        配置好的模型
    """
    if model_name == 'mobilenetv2':
        return create_mobilenetv2_model(num_classes, pretrained)
    elif model_name == 'shufflenetv2':
        size = kwargs.get('size', '1.0x')
        return create_shufflenetv2_model(num_classes, pretrained, size)
    elif model_name == 'squeezenet':
        version = kwargs.get('version', '1_1')
        return create_squeezenet_model(num_classes, pretrained, version)
    elif model_name == 'mobilenetv3_small':
        return create_mobilenetv3_small_model(num_classes, pretrained)
    elif model_name in ['tinyconvnet', 'micronet']:
        # 自定义模型没有预训练权重
        return create_custom_tiny_model(num_classes, model_name)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

# 用于测试模型结构
if __name__ == "__main__":
    # 创建模型
    model = create_model(model_name='mobilenetv3_small', num_classes=10)
    
    # 打印模型结构
    print(model)
    
    # 生成一个随机输入并检查输出尺寸
    dummy_input = torch.randn(1, 3, 224, 224)  # 批次大小为1，3通道，224x224尺寸
    output = model(dummy_input)
    
    print(f"模型输入形状: {dummy_input.shape}")
    print(f"模型输出形状: {output.shape}")
    
    # 打印不同模型的参数数量
    models_to_test = ['mobilenetv2', 'shufflenetv2', 'squeezenet', 'mobilenetv3_small']
    for m_name in models_to_test:
        model = create_model(model_name=m_name, num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        print(f"{m_name} 模型参数数量: {params:,}")
    
    # 测试自定义模型
    custom_models = ['tinyconvnet', 'micronet']
    for m_name in custom_models:
        model = create_model(model_name=m_name, num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        print(f"{m_name} 模型参数数量: {params:,}")
        
        # 测试推理
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"{m_name} 输出形状: {output.shape}")