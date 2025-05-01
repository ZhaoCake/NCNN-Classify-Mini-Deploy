#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型定义
from train.model import create_model, TinyConvNet, MicroNet

def main():
    parser = argparse.ArgumentParser(description='将复杂模型导出为简单标准格式，解决序列化问题')
    parser.add_argument('--input', required=True, help='输入模型路径')
    parser.add_argument('--output', help='输出模型路径（默认为原路径加上_simple后缀）')
    parser.add_argument('--model-type', help='模型类型，不指定则尝试自动检测')
    parser.add_argument('--num-classes', type=int, default=10, help='类别数量')
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_simple.pth"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 加载原始模型
    print(f"正在加载模型: {args.input}")
    try:
        checkpoint = torch.load(args.input, map_location='cpu')
        
        # 检测模型类型和配置
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_type = checkpoint.get('model_name', args.model_type)
            num_classes = checkpoint.get('num_classes', args.num_classes)
            model_kwargs = checkpoint.get('model_kwargs', {})
            state_dict = checkpoint['state_dict']
        elif hasattr(checkpoint, 'state_dict'):
            # 整个模型对象
            model_type = args.model_type or checkpoint.__class__.__name__.lower()
            num_classes = args.num_classes
            model_kwargs = {}
            state_dict = checkpoint.state_dict()
        else:
            # 假设直接是状态字典
            model_type = args.model_type
            num_classes = args.num_classes
            model_kwargs = {}
            state_dict = checkpoint
            
        # 如果没有指定模型类型，尝试从文件名猜测
        if model_type is None:
            filename = os.path.basename(args.input).lower()
            if 'tinyconvnet' in filename:
                model_type = 'tinyconvnet'
            elif 'micronet' in filename:
                model_type = 'micronet'
            elif 'mobilenetv3_small' in filename:
                model_type = 'mobilenetv3_small'
            elif 'shufflenet' in filename:
                model_type = 'shufflenetv2'
            elif 'squeezenet' in filename:
                model_type = 'squeezenet'
            else:
                model_type = 'mobilenetv2'
                
        print(f"检测到模型类型: {model_type}, 类别数: {num_classes}")
        
        # 创建新模型
        model = create_model(model_name=model_type, num_classes=num_classes, 
                            pretrained=False, **model_kwargs)
        
        # 加载状态字典
        model.load_state_dict(state_dict)
        
        # 保存为简单格式
        torch.save(model.state_dict(), args.output)
        print(f"简化模型已保存到: {args.output}")
        
        # 导出TorchScript格式方便NCNN转换
        script_path = os.path.splitext(args.output)[0] + ".pt"
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(script_path)
        print(f"TorchScript模型已保存到: {script_path}")
        
        print(f"现在你可以使用以下命令转换为NCNN格式:")
        print(f"pnnx {script_path} inputshape=[1,3,224,224]")
        
    except Exception as e:
        print(f"模型导出失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
