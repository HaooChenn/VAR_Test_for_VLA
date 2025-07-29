#!/bin/bash

# VAR (Visual Autoregressive Modeling) 推理运行脚本
# 此脚本用于运行所有深度的VAR模型进行图像生成推理

# 设置脚本在遇到错误时退出
set -e

# 颜色定义，用于美化输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

# 配置参数
PRETRAINED_DIR="./pretrained"      # 预训练模型存放目录
OUTPUT_DIR="./outputs"             # 输出结果目录  
DEVICE="cuda:1"                    # 使用GPU序号为1的显卡
CFG_STRENGTH=3.0                   # Classifier-free guidance强度
NUM_IMAGES=20                      # 每个模型生成的图像数量
SEED=42                            # 随机种子，确保结果可重复
MODEL_DEPTHS=(16 20 24 30)         # 要使用的模型深度列表

# 解析命令行参数（如果提供的话）
while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained_dir)
            PRETRAINED_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cfg_strength)
            CFG_STRENGTH="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --pretrained_dir DIR    预训练模型目录 (默认: ./pretrained)"
            echo "  --output_dir DIR        输出目录 (默认: ./outputs)"
            echo "  --device DEVICE         GPU设备 (默认: cuda:1)"
            echo "  --cfg_strength FLOAT    CFG强度 (默认: 3.0)"
            echo "  --num_images INT        每个模型生成图像数量 (默认: 20)"
            echo "  --seed INT              随机种子 (默认: 42)"
            echo "  --help                  显示此帮助信息"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

print_header "VAR图像生成推理开始"

# 显示配置信息
print_info "推理配置:"
echo "  - 预训练模型目录: $PRETRAINED_DIR"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 使用设备: $DEVICE"
echo "  - CFG强度: $CFG_STRENGTH"
echo "  - 每个模型生成图像数量: $NUM_IMAGES"
echo "  - 随机种子: $SEED"
echo "  - 模型深度: ${MODEL_DEPTHS[*]}"

# 检查环境
print_info "检查运行环境..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    print_error "Python未安装或不在PATH中"
    exit 1
fi

print_success "Python环境检查通过"

# 检查必要的Python包
print_info "检查Python依赖包..."
python -c "
import torch
import torchvision
import numpy as np
import PIL
print('PyTorch版本:', torch.__version__)
print('Torchvision版本:', torchvision.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    print('当前GPU:', torch.cuda.current_device())
" 2>/dev/null || {
    print_error "缺少必要的Python依赖包"
    print_info "请确保已安装: torch, torchvision, numpy, pillow"
    exit 1
}

print_success "Python依赖包检查通过"

# 检查预训练模型目录
print_info "检查预训练模型目录..."
if [[ ! -d "$PRETRAINED_DIR" ]]; then
    print_error "预训练模型目录不存在: $PRETRAINED_DIR"
    exit 1
fi

# 检查VAE模型文件
VAE_MODEL="$PRETRAINED_DIR/vae_ch160v4096z32.pth"
if [[ ! -f "$VAE_MODEL" ]]; then
    print_error "VAE模型文件未找到: $VAE_MODEL"
    print_info "请确保已下载VAE模型文件"
    exit 1
fi

print_success "VAE模型文件检查通过: $VAE_MODEL"

# 检查VAR模型文件
print_info "检查VAR模型文件..."
found_models=()
for depth in "${MODEL_DEPTHS[@]}"; do
    model_file="$PRETRAINED_DIR/var_d${depth}.pth"
    if [[ -f "$model_file" ]]; then
        print_success "找到模型: var_d${depth}.pth"
        found_models+=($depth)
    else
        print_warning "模型文件未找到: var_d${depth}.pth"
    fi
done

if [[ ${#found_models[@]} -eq 0 ]]; then
    print_error "没有找到任何VAR模型文件"
    exit 1
fi

print_success "共找到 ${#found_models[@]} 个VAR模型: ${found_models[*]}"

# 创建输出目录
print_info "创建输出目录..."
mkdir -p "$OUTPUT_DIR"
print_success "输出目录创建完成: $OUTPUT_DIR"

# 检查GPU设备
if [[ "$DEVICE" =~ ^cuda:[0-9]+$ ]]; then
    gpu_id=${DEVICE#cuda:}
    print_info "检查GPU设备 $DEVICE..."
    python -c "
import torch
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    gpu_id = $gpu_id
    if gpu_id < device_count:
        print(f'GPU {gpu_id} 可用')
        torch.cuda.set_device(gpu_id)
        print(f'GPU {gpu_id} 信息: {torch.cuda.get_device_name(gpu_id)}')
        print(f'GPU {gpu_id} 内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB')
    else:
        print(f'GPU {gpu_id} 不可用，共有 {device_count} 个GPU')
        exit(1)
else:
    print('CUDA不可用')
    exit(1)
" || {
        print_error "GPU设备检查失败"
        exit 1
    }
    print_success "GPU设备检查通过"
fi

# 记录开始时间
start_time=$(date +%s)
print_info "推理开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 运行推理脚本
print_header "开始VAR模型推理"

python var_inference.py \
    --pretrained_dir "$PRETRAINED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --cfg_strength "$CFG_STRENGTH" \
    --num_images "$NUM_IMAGES" \
    --seed "$SEED" \
    --depths ${found_models[*]} \
    || {
        print_error "推理脚本执行失败"
        exit 1
    }

# 计算总耗时
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

print_header "推理完成"
print_success "推理结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
printf "${GREEN}[SUCCESS]${NC} 总耗时: "
if [[ $hours -gt 0 ]]; then
    printf "%d小时 " $hours
fi
if [[ $minutes -gt 0 ]]; then
    printf "%d分钟 " $minutes
fi
printf "%d秒\n" $seconds

# 显示结果统计
print_info "生成结果统计:"
echo "  - 使用的模型: ${found_models[*]}"
echo "  - 每个模型生成图像数: $NUM_IMAGES"
echo "  - 总图像数: $((${#found_models[@]} * NUM_IMAGES))"
echo "  - CFG强度: $CFG_STRENGTH"
echo "  - 结果保存位置: $OUTPUT_DIR"

# 显示输出目录结构
print_info "输出目录结构:"
if command -v tree &> /dev/null; then
    tree "$OUTPUT_DIR" -L 2
else
    find "$OUTPUT_DIR" -type d | head -10
fi

print_success "VAR图像生成推理全部完成！"
print_info "你可以在 $OUTPUT_DIR 目录中查看生成的图像"

# 可选：打开输出目录（如果是在图形界面环境中）
if [[ -n "$DISPLAY" ]] && command -v xdg-open &> /dev/null; then
    print_info "尝试打开输出目录..."
    xdg-open "$OUTPUT_DIR" 2>/dev/null || true
fi
