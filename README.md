# 图像分类模型从训练到NCNN部署全流程

> nihui！！赛高！

本项目提供了一个完整的图像分类模型训练与部署解决方案，从数据集准备、模型训练到NCNN模型转换，最终在OrangePi3LTS等ARM设备上部署运行。项目支持自定义数据集和类别数量，可作为通用图像分类方案的参考框架。

本项目支持多种轻量级模型用于图像分类任务（MobileNetV2、MobileNetV3-Small、ShuffleNetV2和SqueezeNet），示例数据集使用CIFAR10，并提供快捷下载脚本。如果自己所使用的数据集与CIFAR10有差别，我们也提供了简单的转换脚本，以及部署代码支持改变类别数量。

---

## 项目结构

```
.
├── README.md           # 项目说明文档
├── deploy/             # 部署相关代码
│   └── models/         # 存放转换后的NCNN模型文件
├── outputs/            # 输出目录
│   ├── models/         # 训练后的模型文件
│   ├── logs/           # 训练日志
│   └── visualizations/ # 训练过程可视化结果
├── scripts/            # 工具脚本
│   ├── convert_to_ncnn.py   # 模型转换脚本
│   ├── export_simpler_model.py  # 模型格式简化工具
│   └── download_cifar10.py  # CIFAR10数据集下载和处理脚本
└── train/              # 训练相关代码
    ├── data_loader.py  # 数据加载模块
    ├── model.py        # 模型定义模块
    └── train.py        # 训练主脚本
```

## 功能模块说明

### 1. 数据集下载与预处理 (`scripts/download_cifar10.py`)
- 自动下载CIFAR10数据集
- 将数据集转换为图像格式并按类别组织
- 自动划分为训练集(80%)、验证集(20%)和测试集

### 2. 模型定义 (`train/model.py`)
- 支持多种轻量级模型架构：
  - **MobileNetV2**：适合中等复杂度的场景
  - **MobileNetV3-Small**：比MobileNetV2更轻量
  - **ShuffleNetV2**：高效的轻量级模型
  - **SqueezeNet**：超轻量级模型
  - **TinyConvNet**：自定义超轻量级模型，仅约0.2M参数
  - **MicroNet**：自定义极简模型，仅约0.01M参数，或许适合微控制器。实际是用来测试的
- 支持使用ImageNet预训练权重（标准模型）
- 支持自定义类别数量

### 3. 数据加载和预处理 (`train/data_loader.py`)
- 图像大小调整到224x224
- 训练集数据增强(随机翻转、旋转和颜色抖动等)
- 图像归一化处理
- 高效的数据并行加载

### 4. 模型训练 (`train/train.py`)
- 完整的训练循环实现
- 验证和测试评估功能
- 自适应学习率调整
- 最佳模型保存
- 训练过程可视化
- 自动导出ONNX格式
- 保存模型元数据以便更好地兼容转换工具

### 5. NCNN模型转换 (`scripts/convert_to_ncnn.py`)
- 将PyTorch模型转换为ONNX格式
- 将PyTorch模型转换为TorchScript格式
- 使用pnnx工具将模型转换为NCNN格式
- 支持配置输入尺寸
- 自动处理各种格式的模型文件
- 智能推断模型架构和参数

### 6. 模型格式简化 (`scripts/export_simpler_model.py`)
- 将复杂模型格式转换为简单标准格式
- 解决序列化和兼容性问题
- 直接导出TorchScript模型供NCNN转换使用
- 帮助解决模块导入错误

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib tqdm pillow pnnx
```

### 1. 下载和准备数据集

```bash
python scripts/download_cifar10.py
```

### 2. 训练模型

```bash
# 使用MobileNetV3-Small（推荐轻量级模型）
python train/train.py --model mobilenetv3_small --pretrained --epochs 30 --batch_size 64

# 使用ShuffleNetV2（更轻量级）
python train/train.py --model shufflenetv2 --shufflenet_size 0.5x --pretrained --epochs 30

# 使用SqueezeNet（超轻量级）
python train/train.py --model squeezenet --squeezenet_version 1_1 --pretrained --epochs 40

# 使用自定义超轻量级模型
python train/train.py --model tinyconvnet --epochs 50 --batch_size 128

# 使用极简型MicroNet（极小型设备）
python train/train.py --model micronet --epochs 100 --batch_size 256 --lr 0.02
```

主要参数说明:
- `--model`: 选择模型架构 ['mobilenetv2', 'mobilenetv3_small', 'shufflenetv2', 'squeezenet', 'tinyconvnet', 'micronet']
- `--pretrained`: 使用ImageNet预训练权重
- `--epochs`: 训练轮数
- `--batch_size`: 批处理大小
- `--lr`: 初始学习率 (默认0.01)
- `--output_dir`: 输出目录 (默认./outputs)
- `--shufflenet_size`: ShuffleNetV2模型的大小 ['0.5x', '1.0x', '1.5x', '2.0x']
- `--squeezenet_version`: SqueezeNet模型版本 ['1_0', '1_1']

### 3. 转换为NCNN模型

#### 直接转换

```bash
# 替换model_name为你训练的模型名称，例如mobilenetv3_small_model.pth
python scripts/convert_to_ncnn.py --model ./outputs/models/model_name.pth
```

#### 如果转换遇到问题，使用模型简化工具

```bash
# 1. 先导出简化版模型
python scripts/export_simpler_model.py --input ./outputs/models/model_name.pth

# 2. 使用生成的TorchScript模型直接转换
pnnx ./outputs/models/model_name_simple.pt inputshape=[1,3,224,224]
```

转换后将生成:
- ONNX格式模型 (`./outputs/models/model_name.onnx`)
- TorchScript格式模型 (`./outputs/models/model_name_torchscript.pt` 或 `./outputs/models/model_name_simple.pt`)
- NCNN格式模型 (`./deploy/models/model_name.param` 和 `.bin`)

## 模型推理

模型训练完成并转换为 NCNN 格式后，可以使用以下方法进行推理。

### Python 推理

项目提供了简洁的 Python 推理脚本，基于 NCNN Python 绑定实现：

```bash
# 基本用法
python deploy/python_inference.py [param_path] [bin_path] [imagepath] [class_names_path(可选)]

# 示例
python deploy/python_inference.py deploy/models/tinyconvnet_model.param \
                               deploy/models/tinyconvnet_model.bin \
                               data/images/test/cat/1.png \
                               outputs/class_names.txt
```

参数说明:
- `param_path`: NCNN 参数文件路径
- `bin_path`: NCNN 二进制权重文件路径  
- `imagepath`: 要推理的图像路径
- `class_names_path`: 类别名称文件路径（可选）

推理脚本会自动进行图像预处理并输出前 5 个预测结果及其置信度。

<!-- ### C++ 推理

对于需要更高性能或集成到其它系统的场景，项目也提供了 C++ 推理实现，详见 `deploy/cpp_inference/` 目录。 -->

## 后续计划

- 添加ARM设备(如OrangePi3LTS)上的C++推理实现
- 添加量化功能以进一步优化模型大小和推理速度
- 支持更多数据增强策略
- 改进转换工具以支持更多自定义模型架构

