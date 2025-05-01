#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import torch
import sys

# 添加上级目录到路径，以便导入模型定义
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型定义
from train.model import TinyConvNet, MicroNet, create_model


def check_pnnx_installation():
    """检查pnnx工具是否安装"""
    try:
        subprocess.run(['pnnx', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def load_model_safely(model_path, device='cpu'):
    """安全地加载模型，处理各种可能的错误情况"""
    print(f"尝试加载模型: {model_path}")
    
    try:
        # 尝试直接加载完整模型
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict) and 'state_dict' in model:
            # 模型以字典方式保存，需要重建
            print("检测到模型以字典格式保存，正在重建模型...")
            
            model_name = model.get('model_name', 'mobilenetv2')
            num_classes = model.get('num_classes', 10)
            model_kwargs = model.get('model_kwargs', {})
            
            # 创建模型实例
            net = create_model(model_name=model_name, num_classes=num_classes, 
                               pretrained=False, **model_kwargs)
            
            # 加载状态字典
            net.load_state_dict(model['state_dict'])
            return net
        else:
            # 模型是直接的对象
            return model
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"加载模型时出错: {e}")
        
        # 从文件名猜测模型类型
        model_filename = os.path.basename(model_path).lower()
        state_dict = None
        
        try:
            # 尝试仅加载状态字典
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # 尝试获取模型配置
                    model_name = checkpoint.get('model_name', None)
                    num_classes = checkpoint.get('num_classes', 10)
                    model_kwargs = checkpoint.get('model_kwargs', {})
                else:
                    # 假设整个字典就是状态字典
                    state_dict = checkpoint
        except Exception as e2:
            print(f"加载状态字典时出错: {e2}")
            state_dict = None
        
        # 如果没有从checkpoint获取到model_name，从文件名推断
        if state_dict is None or model_name is None:
            if 'tinyconvnet' in model_filename:
                model_name = 'tinyconvnet'
            elif 'micronet' in model_filename:
                model_name = 'micronet'
            elif 'mobilenetv3_small' in model_filename:
                model_name = 'mobilenetv3_small'
            elif 'shufflenet' in model_filename:
                model_name = 'shufflenetv2'
                # 从文件名确定大小
                if '0.5x' in model_filename:
                    model_kwargs = {'size': '0.5x'}
                elif '2.0x' in model_filename:
                    model_kwargs = {'size': '2.0x'}
                elif '1.5x' in model_filename:
                    model_kwargs = {'size': '1.5x'}
                else:
                    model_kwargs = {'size': '1.0x'}
            elif 'squeezenet' in model_filename:
                model_name = 'squeezenet'
                # 从文件名确定版本
                model_kwargs = {'version': '1_1' if '1_1' in model_filename else '1_0'}
            else:
                model_name = 'mobilenetv2'
                model_kwargs = {}
        
        # 创建空模型
        print(f"创建新的 {model_name} 模型...")
        net = create_model(model_name=model_name, num_classes=num_classes, 
                           pretrained=False, **model_kwargs)
        
        # 如果有状态字典，加载它
        if state_dict is not None:
            print("加载状态字典...")
            try:
                net.load_state_dict(state_dict)
            except Exception as e3:
                print(f"加载状态字典失败: {e3}")
                print("使用未初始化的模型继续...")
        
        return net


def convert_model_to_onnx(model_path, output_path, input_size=(1, 3, 224, 224)):
    """将PyTorch模型转换为ONNX格式"""
    print(f"从 {model_path} 加载PyTorch模型...")
    
    # 使用安全加载函数
    model = load_model_safely(model_path)
    model.eval()
    
    # 生成随机输入
    device = torch.device('cpu')
    dummy_input = torch.randn(*input_size).to(device)
    
    # 导出为ONNX
    print(f"正在导出为ONNX格式: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=12
    )
    print(f"ONNX模型已保存到 {output_path}")
    return output_path


def convert_to_torchscript(model, save_path, input_size=(1, 3, 224, 224)):
    """将 PyTorch 模型转换为 TorchScript 格式"""
    print(f"将模型转换为 TorchScript 格式: {save_path}")
    
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # 创建样例输入
    dummy_input = torch.randn(*input_size).to(device)
    
    # 使用 trace 模式转换为 TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # 保存 TorchScript 模型
    traced_model.save(save_path)
    print(f"TorchScript 模型已保存到 {save_path}")
    return save_path


def convert_onnx_to_ncnn(onnx_path, output_dir):
    """使用pnnx将ONNX模型转换为NCNN格式"""
    # 检查pnnx工具是否安装
    if not check_pnnx_installation():
        print("错误: 未找到pnnx工具。请先安装pnnx:")
        print("参考: https://github.com/pnnx/pnnx 或 https://github.com/Tencent/ncnn/tree/master/tools/pnnx")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"确保输出目录存在: {output_dir}")
    
    # 生成输出文件路径
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    output_param = os.path.join(output_dir, f"{model_name}.param")
    output_bin = os.path.join(output_dir, f"{model_name}.bin")
    
    print(f"正在将 {onnx_path} 转换为NCNN格式...")
    
    # 从ONNX路径推导原始PT模型路径
    pt_path = onnx_path.replace(".onnx", ".pth")
    
    # 首先将模型转换为 TorchScript 格式
    torchscript_path = os.path.join(os.path.dirname(pt_path), f"{model_name}_torchscript.pt")
    
    if os.path.exists(pt_path):
        # 加载原始 PyTorch 模型
        print(f"使用原始PyTorch模型 {pt_path} 进行转换...")
        
        try:
            # 使用安全加载函数加载模型
            model = load_model_safely(pt_path)
            model.eval()
            
            # 转换为 TorchScript 格式
            convert_to_torchscript(model, torchscript_path)
        except Exception as e:
            print(f"转换 PyTorch 模型到 TorchScript 失败: {e}")
            return False
        
        # 使用 pnnx 转换 TorchScript 模型为 NCNN
        if os.path.exists(torchscript_path):
            print(f"使用 TorchScript 模型进行 PNNX 转换...")
            
            # 构建 pnnx 命令
            cmd = [
                'pnnx', 
                torchscript_path, 
                f'ncnnparam={output_param}', 
                f'ncnnbin={output_bin}',
                'inputshape=[1,3,224,224]',
                'fp16=1'
            ]
            
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if process.returncode != 0:
                print("pnnx 转换失败:")
                print(process.stderr.decode())
                return False
        else:
            print(f"错误: 无法创建 TorchScript 模型")
            return False
    else:
        # 如果找不到原始PyTorch模型，尝试使用onnx2ncnn工具
        print(f"找不到原始PyTorch模型，尝试使用onnx2ncnn工具...")
        
        try:
            # 运行onnx2ncnn命令
            cmd = ['onnx2ncnn', onnx_path, output_param, output_bin]
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if process.returncode != 0:
                print("onnx2ncnn转换失败:")
                print(process.stderr.decode())
                return False
                
        except FileNotFoundError:
            print("错误: onnx2ncnn工具未找到")
            print("请确保已安装ncnn并将其工具目录添加到PATH中")
            print("参考: https://github.com/Tencent/ncnn/wiki/how-to-build")
            return False
    
    print(f"NCNN模型已保存到:")
    print(f"  参数文件: {output_param}")
    print(f"  二进制权重: {output_bin}")
    return True


def main():
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为NCNN格式')
    parser.add_argument('--model', default='./outputs/mobilenetv2_cifar10.pth', 
                        help='PyTorch模型路径')
    parser.add_argument('--onnx', default=None, 
                        help='ONNX模型保存路径(默认根据输入模型名生成)')
    parser.add_argument('--ncnn-dir', default='./deploy/models', 
                        help='NCNN模型保存目录')
    parser.add_argument('--width', type=int, default=224, 
                        help='输入图像宽度')
    parser.add_argument('--height', type=int, default=224, 
                        help='输入图像高度')
    parser.add_argument('--channels', type=int, default=3, 
                        help='输入图像通道数')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='类别数量')
    args = parser.parse_args()
    
    # 如果没有指定ONNX输出路径，根据模型路径生成
    if args.onnx is None:
        model_base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.onnx = os.path.join(os.path.dirname(args.model), f"{model_base_name}.onnx")
    
    # 确保输出目录存在
    onnx_dir = os.path.dirname(args.onnx)
    if onnx_dir and not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir, exist_ok=True)
        print(f"创建ONNX输出目录: {onnx_dir}")
    
    # 确保NCNN模型输出目录存在
    os.makedirs(args.ncnn_dir, exist_ok=True)
    print(f"确保NCNN模型输出目录存在: {args.ncnn_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.model):
        if os.path.exists(args.onnx):
            print(f"PyTorch模型 {args.model} 不存在，但找到了ONNX模型，将直接使用它")
        else:
            print(f"错误: PyTorch模型 {args.model} 不存在")
            return
    
    # 如果需要，将模型转换为ONNX
    if not os.path.exists(args.onnx) and os.path.exists(args.model):
        input_size = (1, args.channels, args.height, args.width)
        convert_model_to_onnx(args.model, args.onnx, input_size)
    
    # 将ONNX转换为NCNN (实际上现在主要是先转为TorchScript再到NCNN)
    if os.path.exists(args.onnx):
        success = convert_onnx_to_ncnn(args.onnx, args.ncnn_dir)
        if success:
            print("模型转换完成!")
        else:
            print("模型转换失败!")
    else:
        print(f"错误: ONNX模型 {args.onnx} 不存在")


if __name__ == "__main__":
    main()