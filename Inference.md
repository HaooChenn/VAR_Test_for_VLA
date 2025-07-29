# VAR (Visual Autoregressive Modeling) 推理完整运行指南

## 🚀 快速开始

### 基础运行命令（推荐）
```bash
# 确保在项目根目录，给脚本执行权限
chmod +x run_var_inference.sh

# 使用默认配置运行（CFG=3.0, GPU=cuda:1, 每个模型20张图）
./run_var_inference.sh
```

### 自定义配置运行
```bash
# 完整自定义参数运行
./run_var_inference.sh \
    --pretrained_dir ./pretrained \
    --output_dir ./outputs \
    --device cuda:1 \
    --cfg_strength 3.0 \
    --num_images 20 \
    --seed 42
```

## 📋 命令参数详解

### 必需准备
- **预训练模型文件** 应放在 `./pretrained/` 目录下：
  - `vae_ch160v4096z32.pth` (VAE模型)
  - `var_d16.pth` (VAR-d16模型, 310M参数)
  - `var_d20.pth` (VAR-d20模型, 600M参数)
  - `var_d24.pth` (VAR-d24模型, 1.0B参数)
  - `var_d30.pth` (VAR-d30模型, 2.0B参数)

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained_dir` | `./pretrained` | 预训练模型目录路径 |
| `--output_dir` | `./outputs` | 输出结果目录路径 |
| `--device` | `cuda:1` | 使用的GPU设备（按你要求使用1号GPU） |
| `--cfg_strength` | `3.0` | Classifier-free guidance强度（按你要求设置为3） |
| `--num_images` | `20` | 每个模型生成的图像数量（按你要求生成20张） |
| `--seed` | `42` | 随机种子，确保结果可重复 |
| `--more_smooth` | false | 使用更平滑的生成方式（质量更好但速度更慢） |
| `--no_detailed_stats` | false | 不保存详细的时间统计信息 |

## 🎯 不同使用场景的命令

### 1. 标准推理（按你的要求）
```bash
# 使用1号GPU，CFG强度3.0，每个模型生成20张图片
./run_var_inference.sh --device cuda:1 --cfg_strength 3.0 --num_images 20
```

### 2. 高质量生成（速度较慢）
```bash
# 启用平滑生成，获得更好的视觉质量
./run_var_inference.sh --more_smooth --cfg_strength 4.0
```

### 3. 快速测试（少量图片）
```bash
# 每个模型只生成5张图片进行快速测试
./run_var_inference.sh --num_images 5 --seed 123
```

### 4. 指定特定模型深度
```bash
# 只运行深度为16和24的模型
python var_inference.py \
    --pretrained_dir ./pretrained \
    --output_dir ./outputs \
    --device cuda:1 \
    --cfg_strength 3.0 \
    --num_images 20 \
    --depths 16 24
```

### 5. 使用其他GPU设备
```bash
# 使用0号GPU（如果1号GPU不可用）
./run_var_inference.sh --device cuda:0
```

## 📊 时间测量功能详解

### 新增的精确时间测量特性

1. **GPU同步计时**：使用 `torch.cuda.synchronize()` 确保测量准确性
2. **模型预热**：自动进行3次预热迭代，消除初始化开销
3. **逐张计时**：为每张图像独立测量生成时间
4. **详细统计**：生成多种格式的性能报告

### 输出的时间数据文件

```
outputs/
├── var_d16_cfg3.0/
│   ├── 00_golden_retriever_0.842s.png     # 文件名包含生成时间
│   ├── 01_tabby_cat_0.756s.png
│   ├── ...
│   ├── timing_stats_var_d16.csv           # 详细时间统计CSV
│   ├── model_performance_var_d16.json     # 模型性能JSON报告
│   └── grid_var_d16_with_timing.png       # 带时间标注的网格图
├── var_d20_cfg3.0/
│   └── ...（类似结构）
├── var_d24_cfg3.0/
│   └── ...
├── var_d30_cfg3.0/
│   └── ...
└── performance_comparison.json            # 跨模型性能比较
```

## 🔍 结果分析

### 1. 查看时间统计CSV文件
```bash
# 查看某个模型的详细时间统计
cat outputs/var_d16_cfg3.0/timing_stats_var_d16.csv
```

CSV文件包含：
- `image_id`: 图像编号
- `scene_name`: 场景名称
- `class_id`: ImageNet类别ID
- `generation_time`: 精确生成时间（秒）
- `seed`: 使用的随机种子
- `file_path`: 生成图像的文件路径

### 2. 查看模型性能JSON报告
```bash
# 查看模型性能摘要
cat outputs/var_d16_cfg3.0/model_performance_var_d16.json
```

JSON报告包含：
- 平均生成时间
- 最快/最慢生成时间
- 时间标准差（稳定性指标）
- 吞吐量（图像/秒）
- 预热时间

### 3. 查看跨模型性能比较
```bash
# 查看所有模型的性能对比
cat outputs/performance_comparison.json
```

### 4. 使用快速查看脚本
```bash
# 运行完成后会自动生成快速查看脚本
cd outputs
./quick_view.sh
```

## 🎨 生成的场景类别

脚本会为每个模型生成20张不同场景的图像：

| 类别编号 | 场景名称 | 描述 |
|----------|----------|------|
| 207 | golden_retriever | 金毛寻回犬 |
| 281 | tabby_cat | 虎斑猫 |
| 277 | red_fox | 红狐 |
| 323 | monarch_butterfly | 帝王蝶 |
| 985 | daisy | 雏菊 |
| 973 | rose | 玫瑰 |
| 437 | lighthouse | 灯塔 |
| 483 | castle | 城堡 |
| 500 | cottage | 小屋 |
| 817 | sports_car | 跑车 |
| 820 | steam_locomotive | 蒸汽机车 |
| 554 | sailboat | 帆船 |
| 403 | aircraft_carrier | 航空母舰 |
| 671 | mountain_bike | 山地自行车 |
| 963 | pizza | 披萨 |
| 949 | strawberry | 草莓 |
| 504 | coffee_mug | 咖啡杯 |
| 889 | violin | 小提琴 |
| 414 | backpack | 背包 |
| 879 | umbrella | 雨伞 |

## ⚠️ 注意事项

### 系统要求
- **GPU内存**: 建议至少8GB VRAM
- **CUDA版本**: 支持PyTorch的CUDA版本
- **磁盘空间**: 至少5GB用于模型和输出结果

### 常见问题解决

1. **GPU内存不足**
   ```bash
   # 减少每批生成的图像数量
   ./run_var_inference.sh --num_images 10
   ```

2. **1号GPU不可用**
   ```bash
   # 自动切换到0号GPU
   ./run_var_inference.sh --device cuda:0
   ```

3. **模型文件缺失**
   ```bash
   # 检查哪些模型文件存在
   ls -la pretrained/var_d*.pth
   ```

## 📈 预期性能指标

基于VAR论文的benchmark结果，你可以期待：

| 模型深度 | 参数量 | 预期FID | 预期生成时间/张 |
|----------|--------|---------|----------------|
| VAR-d16 | 310M | ~3.55 | 0.6-1.0秒 |
| VAR-d20 | 600M | ~2.95 | 0.8-1.2秒 |
| VAR-d24 | 1.0B | ~2.33 | 1.0-1.5秒 |
| VAR-d30 | 2.0B | ~1.97 | 1.2-2.0秒 |

*注：实际时间取决于GPU型号和系统配置*

## 🚀 开始推理

确保你的环境配置正确后，运行以下命令开始VAR推理：

```bash
# 标准推理命令（符合你的所有要求）
./run_var_inference.sh \
    --device cuda:1 \
    --cfg_strength 3.0 \
    --num_images 20 \
    --pretrained_dir ./pretrained \
    --output_dir ./outputs

# 如果需要查看实时输出，可以加上时间戳
./run_var_inference.sh 2>&1 | tee "var_inference_$(date +%Y%m%d_%H%M%S).log"
```

这将生成总共80张高质量图像（4个模型×20张图像），并提供每张图像的精确生成时间数据！