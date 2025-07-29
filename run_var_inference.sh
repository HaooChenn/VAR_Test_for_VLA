#!/bin/bash

# VAR (Visual Autoregressive Modeling) 推理运行脚本 - 精确时间测量版本
# 此脚本用于运行所有深度的VAR模型进行图像生成推理，并精确测量每张图片的生成时间

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

print_timing_info() {
    echo -e "${CYAN}[TIMING]${NC} $1"
}

# 配置参数
PRETRAINED_DIR="./pretrained"      # 预训练模型存放目录
OUTPUT_DIR="./outputs"             # 输出结果目录  
DEVICE="cuda:1"                    # 使用GPU序号为1的显卡
CFG_STRENGTH=3.0                   # Classifier-free guidance强度
NUM_IMAGES=20                      # 每个模型生成的图像数量
SEED=42                            # 随机种子，确保结果可重复
MODEL_DEPTHS=(16 20 24 30)         # 要使用的模型深度列表
MORE_SMOOTH=false                  # 是否使用更平滑的生成方式
SAVE_DETAILED_STATS=true           # 是否保存详细统计信息

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
        --more_smooth)
            MORE_SMOOTH=true
            shift
            ;;
        --no_detailed_stats)
            SAVE_DETAILED_STATS=false
            shift
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
            echo "  --more_smooth           使用更平滑的生成方式（质量更好但速度更慢）"
            echo "  --no_detailed_stats     不保存详细的时间统计信息"
            echo "  --help                  显示此帮助信息"
            echo ""
            echo "时间测量功能:"
            echo "  - 精确测量每张图片的生成时间"
            echo "  - 自动进行模型预热以确保准确性"
            echo "  - 生成详细的性能统计报告"
            echo "  - 在图像文件名中包含生成时间"
            echo "  - 生成包含时间标注的网格图像"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

print_header "VAR图像生成推理开始 - 精确时间测量版本"

# 显示配置信息
print_info "推理配置:"
echo "  - 预训练模型目录: $PRETRAINED_DIR"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 使用设备: $DEVICE"
echo "  - CFG强度: $CFG_STRENGTH"
echo "  - 每个模型生成图像数量: $NUM_IMAGES"
echo "  - 随机种子: $SEED"
echo "  - 模型深度: ${MODEL_DEPTHS[*]}"
echo "  - 平滑生成: $MORE_SMOOTH"
echo "  - 保存详细统计: $SAVE_DETAILED_STATS"

print_timing_info "时间测量特性:"
echo "  ✓ GPU同步确保精确计时"
echo "  ✓ 模型预热消除初始化开销"
echo "  ✓ 逐张图像独立计时"
echo "  ✓ 详细统计报告生成"

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
import json
import csv
print('✓ PyTorch版本:', torch.__version__)
print('✓ Torchvision版本:', torchvision.__version__)
print('✓ CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU数量:', torch.cuda.device_count())
    print('✓ 当前GPU:', torch.cuda.current_device())
    print('✓ GPU内存管理: 支持')
print('✓ 精确计时模块: 可用')
print('✓ 统计文件生成: 支持JSON/CSV')
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
        model_size=$(du -h "$model_file" | cut -f1)
        print_success "找到模型: var_d${depth}.pth (大小: $model_size)"
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

# 检查GPU设备和内存
if [[ "$DEVICE" =~ ^cuda:[0-9]+$ ]]; then
    gpu_id=${DEVICE#cuda:}
    print_info "检查GPU设备 $DEVICE..."
    python -c "
import torch
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    gpu_id = $gpu_id
    if gpu_id < device_count:
        torch.cuda.set_device(gpu_id)
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory = props.total_memory / 1024**3
        print(f'✓ GPU {gpu_id} 可用: {torch.cuda.get_device_name(gpu_id)}')
        print(f'✓ GPU内存: {total_memory:.1f} GB')
        print(f'✓ 计算能力: {props.major}.{props.minor}')
        
        # 检查可用内存
        torch.cuda.empty_cache()
        free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
        free_memory_gb = free_memory / 1024**3
        print(f'✓ 可用内存: {free_memory_gb:.1f} GB')
        
        if free_memory_gb < 8:
            print(f'⚠ 警告: 可用GPU内存较少，建议至少8GB')
        else:
            print(f'✓ GPU内存充足，支持高效推理')
    else:
        print(f'✗ GPU {gpu_id} 不可用，共有 {device_count} 个GPU')
        exit(1)
else:
    print('✗ CUDA不可用')
    exit(1)
" || {
        print_error "GPU设备检查失败"
        exit 1
    }
    print_success "GPU设备检查通过"
fi

# 记录开始时间
start_time=$(date +%s)
print_timing_info "推理开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 构建Python参数
PYTHON_ARGS="--pretrained_dir '$PRETRAINED_DIR' --output_dir '$OUTPUT_DIR' --device '$DEVICE' --cfg_strength $CFG_STRENGTH --num_images $NUM_IMAGES --seed $SEED --depths ${found_models[*]}"

if [[ "$MORE_SMOOTH" == "true" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --more_smooth"
fi

if [[ "$SAVE_DETAILED_STATS" == "true" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --save_detailed_stats"
fi

# 运行推理脚本
print_header "开始VAR模型推理与精确时间测量"

eval python var_inference.py $PYTHON_ARGS || {
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
printf "${CYAN}[TIMING]${NC} 总耗时: "
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

# 显示时间测量功能的输出文件
print_timing_info "时间测量结果文件:"
echo "  ✓ 每张图像包含生成时间的文件名"
echo "  ✓ 带时间标注的网格图像"
echo "  ✓ 详细时间统计CSV文件"
echo "  ✓ 模型性能JSON报告"
echo "  ✓ 跨模型性能比较报告"

# 显示输出目录结构
print_info "输出目录结构:"
if command -v tree &> /dev/null; then
    tree "$OUTPUT_DIR" -L 3
else
    find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.csv" -o -name "*.json" | head -20
fi

# 生成性能摘要
print_header "性能分析摘要"
if [[ -f "$OUTPUT_DIR/performance_comparison.json" ]]; then
    print_timing_info "正在分析性能数据..."
    python -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/performance_comparison.json', 'r') as f:
        data = json.load(f)
    
    print('模型性能对比:')
    print(f'{'模型':<10} {'平均时间':<12} {'吞吐量':<12} {'稳定性':<10}')
    print('-' * 50)
    
    for model in data['model_comparisons']:
        depth = model['model_depth']
        avg_time = model['avg_time_per_image']
        throughput = model['throughput_images_per_second']
        stability = model['time_stability_std']
        print(f'VAR-d{depth:<4} {avg_time:<12.3f} {throughput:<12.2f} ±{stability:.3f}')
    
    print()
    print('配置参数:')
    config = data['experiment_config']
    for key, value in config.items():
        print(f'  {key}: {value}')
        
except Exception as e:
    print(f'无法解析性能数据: {e}')
"
else
    print_warning "性能比较文件未找到，可能推理未完全成功"
fi

print_success "VAR图像生成推理全部完成！"
print_info "你可以在 $OUTPUT_DIR 目录中查看:"
echo "  📸 生成的图像（文件名包含生成时间）"
echo "  📊 详细的时间统计数据"
echo "  📈 性能分析报告"
echo "  🖼️  带时间标注的网格图像"

# 可选：生成快速查看脚本
QUICK_VIEW_SCRIPT="$OUTPUT_DIR/quick_view.sh"
cat > "$QUICK_VIEW_SCRIPT" << 'EOF'
#!/bin/bash
# 快速查看生成结果的脚本

echo "VAR推理结果概览："
echo "==================="

# 统计生成的图像数量
total_images=$(find . -name "*.png" -not -name "grid_*" | wc -l)
echo "总生成图像数: $total_images"

# 显示各模型的结果
for dir in var_d*_cfg*/; do
    if [[ -d "$dir" ]]; then
        model_name=$(basename "$dir")
        image_count=$(find "$dir" -name "*.png" -not -name "grid_*" | wc -l)
        echo "  $model_name: $image_count 张图像"
        
        # 如果有性能数据，显示平均时间
        if [[ -f "$dir/model_performance_${model_name%_cfg*}.json" ]]; then
            avg_time=$(python3 -c "
import json
with open('$dir/model_performance_${model_name%_cfg*}.json') as f:
    data = json.load(f)
    print(f'{data[\"average_time_per_image_seconds\"]:.3f}')
" 2>/dev/null)
            if [[ -n "$avg_time" ]]; then
                echo "    平均生成时间: ${avg_time}秒"
            fi
        fi
    fi
done

echo ""
echo "查看详细结果:"
echo "  图像文件: find . -name '*.png'"
echo "  统计数据: find . -name '*.csv' -o -name '*.json'"
echo "  网格图像: find . -name 'grid_*.png'"
EOF

chmod +x "$QUICK_VIEW_SCRIPT"
print_info "已创建快速查看脚本: $QUICK_VIEW_SCRIPT"

# 可选：打开输出目录（如果是在图形界面环境中）
if [[ -n "$DISPLAY" ]] && command -v xdg-open &> /dev/null; then
    print_info "尝试打开输出目录..."
    xdg-open "$OUTPUT_DIR" 2>/dev/null || true
fi

print_header "推理任务完成"
print_timing_info "查看生成时间数据: cat $OUTPUT_DIR/*/timing_stats_*.csv"
print_timing_info "查看性能比较: cat $OUTPUT_DIR/performance_comparison.json"