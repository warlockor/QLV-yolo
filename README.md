# QLV-yolo (YOLO26n INT8 on ARM Mali Bifrost)

本仓库提供 3 种 C++ 部署方案，目标硬件为 Linux Nano-PC + Arm Mali Bifrost（arch 10.8.6）。

## 三种方案

1. `Arm NN + OpenCL(GpuAcc)`（推荐，适配 Mali）
2. `TensorFlow Lite CPU`（兼容性最好）
3. `TensorFlow Lite GPU Delegate`（Linux ARM 上为实验性方案）

## 目录结构

```text
.
├── Makefile
└── src
    ├── main.cpp                 # Arm NN + OpenCL
    ├── tflite_cpu_main.cpp      # TFLite CPU
    └── tflite_gpu_main.cpp      # TFLite GPU Delegate
```

## 通用前置依赖（所有方案都要）

```bash
sudo apt update
sudo apt install -y \
  build-essential make pkg-config git \
  libopencv-dev \
  ocl-icd-opencl-dev opencl-headers clinfo \
  libprotobuf-dev protobuf-compiler \
  libboost-all-dev
```

## 方案一：Arm NN + OpenCL(GpuAcc)

### 需要的额外依赖

- `armnn`（`libarmnn.so`）
- `armnnTfLiteParser`（`libarmnnTfLiteParser.so`）
- 头文件：
  - `armnn/ArmNN.hpp`
  - `armnnTfLiteParser/ITfLiteParser.hpp`

### 安装步骤

1. 安装 Arm NN（包管理器或源码编译安装）。
2. 确认 OpenCL 可用：`clinfo` 能看到 Mali 设备。
3. 若非系统默认路径，编译时通过 `make` 传入库路径变量。

### Arm NN 安装细化（推荐）

在 ARM 板卡上，Arm NN 常用“源码编译安装”。最关键是安装出：

- 头文件：`armnn/ArmNN.hpp`
- 头文件：`armnnTfLiteParser/ITfLiteParser.hpp`
- 库：`libarmnn.so`
- 库：`libarmnnTfLiteParser.so`

建议流程（示例）：

```bash
# 1) 获取源码
git clone https://github.com/ARM-software/armnn.git
cd armnn

# 2) 按官方脚本准备第三方依赖并构建
#    实际命令会随版本变化，请以官方文档为准

# 3) 安装到系统目录或自定义前缀（例如 /opt/armnn）
#    安装完成后确认 include/ 与 lib/ 下有对应文件
```

官方参考：

- Arm NN GitHub: [https://github.com/ARM-software/armnn](https://github.com/ARM-software/armnn)
- Arm NN 文档（安装/构建）: [https://arm-software.github.io/armnn/latest/](https://arm-software.github.io/armnn/latest/)

```bash
make armnn \
  ARMNN_INCLUDE_DIR=/your/path/include \
  ARMNN_TFLITE_INCLUDE_DIR=/your/path/include \
  ARMNN_LIB_DIR=/your/path/lib
```

运行：

```bash
./build/yolo_armnn_opencl \
  --model yolo26n_int8.tflite \
  --image test.jpg \
  --output result_armnn.jpg
```

## 方案二：TensorFlow Lite CPU

### 需要的额外依赖

- `TensorFlow Lite C++` 库（例如 `libtensorflow-lite.so`）
- 头文件（如）：
  - `tensorflow/lite/interpreter.h`
  - `tensorflow/lite/model.h`

### 安装步骤

1. 安装/编译 TensorFlow Lite C++（系统包或源码编译）。
2. 确认库和头文件可被 `make` 找到（必要时显式指定路径）。

### TensorFlow Lite C++ 安装细化（推荐）

很多 ARM Linux 发行版仓库没有完整的 TFLite C++ 开发包，常见做法是“源码编译并安装”：

```bash
# 1) 获取 TensorFlow 源码（含 Lite）
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# 2) 用官方推荐方式构建 TensorFlow Lite C++ 库（Bazel 或 CMake）
#    实际命令与版本相关，请参考官方文档

# 3) 安装/拷贝头文件和库到目标目录（如 /usr/local 或 /opt/tflite）
#    需要保证至少有：
#    - tensorflow/lite/interpreter.h
#    - tensorflow/lite/model.h
#    - libtensorflow-lite.so
```

官方参考：

- TensorFlow Lite 指南: [https://www.tensorflow.org/lite/guide](https://www.tensorflow.org/lite/guide)
- TFLite C++ 推理: [https://www.tensorflow.org/lite/guide/inference](https://www.tensorflow.org/lite/guide/inference)
- TFLite CMake 构建: [https://www.tensorflow.org/lite/guide/build_cmake](https://www.tensorflow.org/lite/guide/build_cmake)

```bash
make tflite_cpu \
  TFLITE_INCLUDE_DIR=/your/path/include \
  TFLITE_LIB_DIR=/your/path/lib \
  TFLITE_LIB_NAME=tensorflow-lite
```

运行：

```bash
./build/yolo_tflite_cpu \
  --model yolo26n_int8.tflite \
  --image test.jpg \
  --output result_tflite_cpu.jpg
```

## 方案三：TensorFlow Lite GPU Delegate（实验性）

> 说明：在 Linux ARM + Mali 上，该方案通常不如 Arm NN + OpenCL 稳定，仅建议测试验证。

### 需要的额外依赖

- TensorFlow Lite C++ 基础库（同方案二）
- TFLite GPU delegate 库（例如 `libtensorflowlite_gpu_delegate.so`）
- 头文件：
  - `tensorflow/lite/delegates/gpu/delegate.h`

### 安装步骤

1. 准备 TensorFlow Lite C++ 与 GPU delegate 库。
2. 若 `ModifyGraphWithDelegate` 失败，说明当前系统/驱动/构建组合不支持该 delegate。

### GPU Delegate 额外说明

- 你除了 `libtensorflow-lite.so`，还需要 GPU delegate 库（如 `libtensorflowlite_gpu_delegate.so`）。
- Linux ARM + Mali 下该路线受版本/驱动影响较大，能编过不代表一定能运行成功。
- 如果 GPU delegate 不稳定，建议回到方案一（Arm NN + OpenCL）作为主线。

参考：

- TFLite GPU Delegate 文档: [https://www.tensorflow.org/lite/performance/gpu](https://www.tensorflow.org/lite/performance/gpu)

```bash
make tflite_gpu \
  TFLITE_INCLUDE_DIR=/your/path/include \
  TFLITE_LIB_DIR=/your/path/lib \
  TFLITE_LIB_NAME=tensorflow-lite \
  TFLITE_GPU_LIB_NAME=tensorflowlite_gpu_delegate
```

运行：

```bash
./build/yolo_tflite_gpu_delegate \
  --model yolo26n_int8.tflite \
  --image test.jpg \
  --output result_tflite_gpu.jpg
```

## 模型格式与参数（支持 YOLO-Seg）

- 模型格式：`INT8/FP16/FP32 .tflite`（例如 `yolo26n-seg_armnn_int8.tflite`）
- 当前后处理按常见 YOLO 输出假设：`[1, N, 4 + 1 + num_classes]`
- 当前代码已支持 `yolo*-seg` 常见双输出（检测头 + mask proto），会在结果图叠加分割掩码。
- 若导出布局不同，请按你的输出张量调整源码中的后处理解析。

所有可执行程序使用相同参数：

- `--model`：模型路径
- `--image`：输入图片路径
- `--output`：输出图片路径
- `--classes`：类别数（默认 `80`）
- `--conf`：置信度阈值（默认 `0.25`）
- `--iou`：NMS IoU 阈值（默认 `0.45`）

## 从 `.pt` 导出三种方案可读模型

本仓库提供脚本：`tools/export_ultralytics_pt.py`  
输入 Ultralytics `.pt`，自动导出三种方案可直接使用的模型文件：

- 方案一（Arm NN）：`*_armnn_int8.tflite`
- 方案二（TFLite CPU）：`*_tflite_cpu_int8.tflite`
- 方案三（TFLite GPU Delegate）：`*_tflite_gpu_fp16.tflite`（若 FP16 失败自动回退 `*_tflite_gpu_fp32.tflite`）

### 导出脚本依赖（Python）

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics tensorflow
```

> 说明：不同 Ultralytics/TensorFlow 版本对 TFLite 导出依赖略有差异，若提示缺包，请按报错补装相关包。

### 使用方式

```bash
python3 tools/export_ultralytics_pt.py \
  --pt /path/to/yolo26n.pt \
  --out-dir models \
  --imgsz 640 \
  --data /path/to/dataset.yaml
```

参数说明：

- `--pt`：输入 `.pt` 模型路径
- `--out-dir`：导出目录（默认 `models`）
- `--imgsz`：导出推理尺寸（默认 `640`）
- `--data`：INT8 校准数据配置（必填，需为真实存在的 yaml 路径）
- `--device`：可选，导出设备（如 `cpu`）

## 评估三种导出结果（验证导出效果）

本仓库提供脚本：`tools/eval_exported_models.py`，用于对三种导出模型做统一评估并输出对比结果。

### 评估脚本依赖（Python）

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics tensorflow
```

### 方法一：自动发现模型（推荐）

当你已经通过导出脚本生成了：

- `models/<stem>_armnn_int8.tflite`
- `models/<stem>_tflite_cpu_int8.tflite`
- `models/<stem>_tflite_gpu_fp16.tflite`（或 fp32）

可直接运行：

```bash
python3 tools/eval_exported_models.py \
  --data /abs/path/data.yaml \
  --out-dir models \
  --stem yolo26n \
  --imgsz 640 \
  --task segment \
  --split val \
  --device cpu \
  --batch 1 \
  --save-json reports/eval_three_models.json
```

### 方法二：手动指定三个模型路径

```bash
python3 tools/eval_exported_models.py \
  --data /abs/path/data.yaml \
  --armnn-model /abs/path/yolo26n_armnn_int8.tflite \
  --tflite-cpu-model /abs/path/yolo26n_tflite_cpu_int8.tflite \
  --tflite-gpu-model /abs/path/yolo26n_tflite_gpu_fp16.tflite \
  --imgsz 640 \
  --task segment \
  --split val \
  --device cpu
```

### 输出内容

脚本会打印并可选保存以下指标（根据 `--task`）：

- `box_mAP50`、`box_mAP50-95`
- `seg_mAP50`、`seg_mAP50-95`（`--task segment` 时）
- `box_precision/recall` 与 `seg_precision/recall`
- `speed(ms)` 与估算 `FPS`

用于快速验证：

- 导出后精度是否明显下降
- 三种导出结果在同一数据集上的速度差异

## 常用 Make 命令

```bash
make help          # 查看可用 target
make all           # 编译三种方案
make clean         # 清理 build 目录
```
