#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import ncnn

def load_class_names(filename):
    """加载类别名称"""
    names = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            names = [line.strip() for line in f.readlines()]
    return names

def print_topk(prob, class_names=None, top_k=5):
    """打印前k个预测结果"""
    indices = np.argsort(-prob)
    
    print("\n===== 预测结果 =====")
    for i in range(min(top_k, len(prob))):
        idx = indices[i]
        name = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
        print(f"{i+1}. {name}: {prob[idx]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} [param_path] [bin_path] [imagepath] [class_names_path(可选)]")
        print(f"例如: {sys.argv[0]} model.param model.bin test.jpg class_names.txt")
        sys.exit(0)

    param_path = sys.argv[1]
    bin_path = sys.argv[2]
    imagepath = sys.argv[3]
    class_names_path = sys.argv[4] if len(sys.argv) > 4 else None

    # 加载图片
    img = cv2.imread(imagepath)
    if img is None:
        print(f"cv2.imread {imagepath} 失败\n")
        sys.exit(0)

    # 加载类别名称
    class_names = []
    if class_names_path:
        class_names = load_class_names(class_names_path)

    # 创建NCNN模型并加载参数和权重
    net = ncnn.Net()
    net.opt.num_threads = 4
    net.load_param(param_path)
    net.load_model(bin_path)
    
    # 预处理图像
    img_resized = cv2.resize(img, (224, 224))
    mean_vals = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]
    norm_vals = [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)]
    mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 224, 224)
    mat_in.substract_mean_normalize(mean_vals, norm_vals)
    
    # 推理
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")
    
    # 处理输出
    prob = np.array(mat_out)
    
    # 如果需要应用softmax
    if np.max(prob) > 1.0 or np.min(prob) < 0.0:
        prob = np.exp(prob - np.max(prob))
        prob = prob / np.sum(prob)
    
    # 打印结果
    print_topk(prob, class_names, 5)