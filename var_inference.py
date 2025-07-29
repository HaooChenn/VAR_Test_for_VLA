#!/usr/bin/env python3
"""
VAR (Visual Autoregressive Modeling) 推理脚本 - 增强时间测量版本
支持多种深度模型，生成高质量图像，并精确测量每张图片的生成时间
"""

import os
import sys
import time
import random
import argparse
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
import torchvision
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw

# 禁用默认参数初始化以加速
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var

@dataclass
class ImageGenerationStats:
    """存储单张图像生成的详细统计信息"""
    image_id: int
    scene_name: str
    class_id: int
    model_depth: int
    cfg_strength: float
    generation_time: float  # 单张图像生成时间（秒）
    seed: int
    file_path: str

@dataclass
class ModelPerformanceStats:
    """存储模型整体性能统计信息"""
    model_depth: int
    total_images: int
    total_time: float
    average_time_per_image: float
    min_time: float
    max_time: float
    std_time: float
    warmup_time: float  # 模型预热时间
    actual_inference_time: float  # 实际推理时间（排除预热）

class PreciseTimer:
    """精确的GPU时间测量工具"""
    
    def __init__(self, device: str):
        self.device = device
        self.use_cuda = device.startswith('cuda')
        
    def __enter__(self):
        """进入计时上下文"""
        if self.use_cuda:
            torch.cuda.synchronize()  # 确保之前的操作都完成
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出计时上下文"""
        if self.use_cuda:
            torch.cuda.synchronize()  # 确保当前操作完成
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

class VARInferenceEngine:
    """VAR推理引擎，支持多种模型深度和精确时间测量"""
    
    def __init__(self, device: str = 'cuda:1'):
        self.device = device
        self.models = {}  # 存储不同深度的模型
        self.vae = None
        self.timer = PreciseTimer(device)
        
        # ImageNet类别到场景的映射 - 精选20个有代表性的场景类别
        self.scene_classes = {
            'golden_retriever': 207,       # 金毛寻回犬
            'tabby_cat': 281,             # 虎斑猫  
            'red_fox': 277,               # 红狐
            'monarch_butterfly': 323,      # 帝王蝶
            'daisy': 985,                 # 雏菊
            'rose': 973,                  # 玫瑰
            'lighthouse': 437,            # 灯塔
            'castle': 483,                # 城堡
            'cottage': 500,               # 小屋
            'sports_car': 817,            # 跑车
            'steam_locomotive': 820,       # 蒸汽机车
            'sailboat': 554,              # 帆船
            'aircraft_carrier': 403,       # 航空母舰
            'mountain_bike': 671,         # 山地自行车
            'pizza': 963,                 # 披萨
            'strawberry': 949,            # 草莓
            'coffee_mug': 504,            # 咖啡杯
            'violin': 889,                # 小提琴
            'backpack': 414,              # 背包
            'umbrella': 879               # 雨伞
        }
        
        self.patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        
    def load_models(self, pretrained_dir: str = './pretrained') -> None:
        """加载所有可用的VAR模型"""
        pretrained_path = Path(pretrained_dir)
        
        # 首先加载VAE模型
        vae_path = pretrained_path / 'vae_ch160v4096z32.pth'
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE模型文件未找到: {vae_path}")
            
        print(f"正在加载VAE模型...")
        self.vae, _ = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=self.device, patch_nums=self.patch_nums,
            num_classes=1000, depth=16, shared_aln=False,
        )
        self.vae.load_state_dict(torch.load(vae_path, map_location='cpu'), strict=True)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        print(f"VAE模型加载完成")
        
        # 加载不同深度的VAR模型
        available_depths = [16, 20, 24, 30]
        for depth in available_depths:
            model_path = pretrained_path / f'var_d{depth}.pth'
            if model_path.exists():
                print(f"正在加载VAR-d{depth}模型...")
                _, var_model = build_vae_var(
                    V=4096, Cvae=32, ch=160, share_quant_resi=4,
                    device=self.device, patch_nums=self.patch_nums,
                    num_classes=1000, depth=depth, shared_aln=False,
                )
                var_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
                var_model.eval()
                for p in var_model.parameters():
                    p.requires_grad_(False)
                
                self.models[depth] = var_model
                print(f"VAR-d{depth}模型加载完成")
            else:
                print(f"VAR-d{depth}模型文件未找到: {model_path}")
        
        if not self.models:
            raise RuntimeError("没有找到任何VAR模型文件")
            
    def setup_inference_environment(self, seed: int = 42) -> None:
        """设置推理环境"""
        # 设置随机种子
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # 设置确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 优化推理性能
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        print(f"推理环境设置完成 - 设备: {self.device}, 种子: {seed}")
        
    def warmup_model(self, var_model, num_warmup_iterations: int = 3) -> float:
        """模型预热，返回预热时间"""
        print(f"正在进行模型预热（{num_warmup_iterations}次迭代）...")
        
        # 创建虚拟输入进行预热
        dummy_label = torch.tensor([0], device=self.device)
        
        with PreciseTimer(self.device) as timer:
            for i in range(num_warmup_iterations):
                with torch.inference_mode():
                    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                        _ = var_model.autoregressive_infer_cfg(
                            B=1,
                            label_B=dummy_label,
                            cfg=1.5,  # 使用较小的CFG值进行预热
                            top_k=900,
                            top_p=0.95,
                            g_seed=42,
                            more_smooth=False
                        )
                # 清理缓存
                torch.cuda.empty_cache()
        
        warmup_time = timer.elapsed_time
        print(f"模型预热完成，耗时: {warmup_time:.3f}秒")
        return warmup_time
    
    def generate_single_image(self, 
                             var_model,
                             class_id: int,
                             cfg_strength: float = 3.0,
                             seed: int = 42,
                             more_smooth: bool = False) -> Tuple[torch.Tensor, float]:
        """
        生成单张图像并精确测量时间
        
        Returns:
            生成的图像张量和生成时间（秒）
        """
        label_tensor = torch.tensor([class_id], device=self.device)
        
        # 精确测量单张图像生成时间
        with PreciseTimer(self.device) as timer:
            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                    generated_image = var_model.autoregressive_infer_cfg(
                        B=1,  # 单张图像生成
                        label_B=label_tensor,
                        cfg=cfg_strength,
                        top_k=900,
                        top_p=0.95,
                        g_seed=seed,
                        more_smooth=more_smooth
                    )
        
        return generated_image[0], timer.elapsed_time  # 返回第一张图像和生成时间
        
    def generate_images_with_timing(self, 
                                   model_depth: int,
                                   cfg_strength: float = 3.0,
                                   num_images: int = 20,
                                   seed: int = 42,
                                   more_smooth: bool = False) -> Tuple[List[torch.Tensor], List[str], List[ImageGenerationStats], ModelPerformanceStats]:
        """
        使用指定深度的模型生成图像，并详细记录每张图像的生成时间
        
        Returns:
            生成的图像列表、场景名称列表、每张图像的统计信息、模型整体性能统计
        """
        if model_depth not in self.models:
            raise ValueError(f"模型深度 {model_depth} 不可用，可用深度: {list(self.models.keys())}")
            
        var_model = self.models[model_depth]
        
        # 选择要生成的类别（循环使用确保有足够的图）
        scene_items = list(self.scene_classes.items())
        selected_scenes = []
        class_labels = []
        
        for i in range(num_images):
            scene_name, class_id = scene_items[i % len(scene_items)]
            selected_scenes.append(scene_name)
            class_labels.append(class_id)
            
        print(f"使用VAR-d{model_depth}生成 {num_images} 张图像")
        print(f"CFG强度: {cfg_strength}, 种子: {seed}")
        print(f"生成类别: {', '.join(selected_scenes[:10])}{'...' if len(selected_scenes) > 10 else ''}")
        
        # 模型预热
        warmup_time = self.warmup_model(var_model)
        
        # 开始逐张生成并计时
        generated_images = []
        image_stats = []
        generation_times = []
        
        print(f"\n开始逐张图像生成（精确计时）:")
        
        total_start_time = time.perf_counter()
        
        for i, (scene_name, class_id) in enumerate(zip(selected_scenes, class_labels)):
            print(f"  生成第 {i+1}/{num_images} 张: {scene_name} (类别ID: {class_id})", end=" -> ")
            
            # 为每张图像设置不同的种子，确保多样性
            image_seed = seed + i
            
            # 生成单张图像并测量时间
            image, gen_time = self.generate_single_image(
                var_model=var_model,
                class_id=class_id,
                cfg_strength=cfg_strength,
                seed=image_seed,
                more_smooth=more_smooth
            )
            
            generation_times.append(gen_time)
            generated_images.append(image)
            
            # 创建图像统计信息
            stat = ImageGenerationStats(
                image_id=i,
                scene_name=scene_name,
                class_id=class_id,
                model_depth=model_depth,
                cfg_strength=cfg_strength,
                generation_time=gen_time,
                seed=image_seed,
                file_path=""  # 稍后会设置
            )
            image_stats.append(stat)
            
            print(f"耗时 {gen_time:.3f}秒")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        total_actual_time = time.perf_counter() - total_start_time
        
        # 计算统计信息
        avg_time = np.mean(generation_times)
        min_time = np.min(generation_times)
        max_time = np.max(generation_times)
        std_time = np.std(generation_times)
        
        model_stats = ModelPerformanceStats(
            model_depth=model_depth,
            total_images=num_images,
            total_time=total_actual_time,
            average_time_per_image=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            warmup_time=warmup_time,
            actual_inference_time=total_actual_time
        )
        
        print(f"\n生成完成统计:")
        print(f"  总时间: {total_actual_time:.3f}秒 (包含预热: {warmup_time:.3f}秒)")
        print(f"  平均每张: {avg_time:.3f}秒")
        print(f"  最快: {min_time:.3f}秒, 最慢: {max_time:.3f}秒")
        print(f"  标准差: {std_time:.3f}秒")
        
        return generated_images, selected_scenes, image_stats, model_stats
        
    def save_results_with_timing(self, 
                                images: List[torch.Tensor],
                                scene_names: List[str],
                                image_stats: List[ImageGenerationStats],
                                model_stats: ModelPerformanceStats,
                                output_dir: str = './outputs') -> None:
        """保存生成结果和详细的时间统计信息"""
        output_path = Path(output_dir)
        model_output_dir = output_path / f'var_d{model_stats.model_depth}_cfg{image_stats[0].cfg_strength}'
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存单独的图像并更新文件路径
        for i, (img, scene_name, stat) in enumerate(zip(images, scene_names, image_stats)):
            filename = f'{i:02d}_{scene_name}_{stat.generation_time:.3f}s.png'
            img_path = model_output_dir / filename
            img_pil = torchvision.transforms.ToPILImage()(img)
            img_pil.save(img_path)
            
            # 更新统计信息中的文件路径
            stat.file_path = str(img_path)
            
        # 创建网格图像，在图像上标注生成时间
        grid_img = torchvision.utils.make_grid(
            torch.stack(images), 
            nrow=5,  # 5列
            padding=4, 
            pad_value=1.0
        )
        grid_pil = torchvision.transforms.ToPILImage()(grid_img)
        
        # 在网格图像上添加时间标注
        draw = PImageDraw.Draw(grid_pil)
        font_size = max(12, grid_pil.width // 80)  # 动态字体大小
        
        # 计算每个图像在网格中的位置并添加时间标注
        img_width = grid_img.shape[2] // 5  # 5列
        img_height = grid_img.shape[1] // ((len(images) + 4) // 5)  # 计算行数
        
        for i, stat in enumerate(image_stats):
            row = i // 5
            col = i % 5
            x = col * img_width + 5
            y = row * img_height + 5
            time_text = f"{stat.generation_time:.3f}s"
            
            # 添加黑色背景的白色文字
            try:
                draw.text((x, y), time_text, fill='white', stroke_width=1, stroke_fill='black')
            except:
                draw.text((x, y), time_text, fill='white')
        
        grid_pil.save(model_output_dir / f'grid_var_d{model_stats.model_depth}_with_timing.png')
        
        # 保存详细的CSV统计文件
        csv_path = model_output_dir / f'timing_stats_var_d{model_stats.model_depth}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_id', 'scene_name', 'class_id', 'generation_time', 'seed', 'file_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for stat in image_stats:
                writer.writerow({
                    'image_id': stat.image_id,
                    'scene_name': stat.scene_name,
                    'class_id': stat.class_id,
                    'generation_time': f"{stat.generation_time:.6f}",
                    'seed': stat.seed,
                    'file_path': stat.file_path
                })
        
        # 保存模型整体性能统计的JSON文件
        json_path = model_output_dir / f'model_performance_var_d{model_stats.model_depth}.json'
        performance_dict = {
            'model_depth': model_stats.model_depth,
            'total_images': model_stats.total_images,
            'total_time_seconds': round(model_stats.total_time, 6),
            'average_time_per_image_seconds': round(model_stats.average_time_per_image, 6),
            'min_time_seconds': round(model_stats.min_time, 6),
            'max_time_seconds': round(model_stats.max_time, 6),
            'std_time_seconds': round(model_stats.std_time, 6),
            'warmup_time_seconds': round(model_stats.warmup_time, 6),
            'actual_inference_time_seconds': round(model_stats.actual_inference_time, 6),
            'images_per_second': round(model_stats.total_images / model_stats.actual_inference_time, 3),
            'cfg_strength': image_stats[0].cfg_strength,
            'seed_range': f"{image_stats[0].seed}-{image_stats[-1].seed}"
        }
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(performance_dict, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"结果已保存至: {model_output_dir}")
        print(f"- 单独图像: {len(images)} 张 (文件名包含生成时间)")
        print(f"- 网格图像: grid_var_d{model_stats.model_depth}_with_timing.png")
        print(f"- 时间统计CSV: timing_stats_var_d{model_stats.model_depth}.csv")
        print(f"- 性能统计JSON: model_performance_var_d{model_stats.model_depth}.json")

def main():
    parser = argparse.ArgumentParser(description='VAR图像生成推理脚本 - 精确时间测量版本')
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained',
                       help='预训练模型目录路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录路径')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='使用的GPU设备')
    parser.add_argument('--cfg_strength', type=float, default=3.0,
                       help='Classifier-free guidance强度')
    parser.add_argument('--num_images', type=int, default=20,
                       help='每个模型生成的图像数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='基础随机种子')
    parser.add_argument('--depths', nargs='+', type=int, default=[16, 20, 24, 30],
                       help='要使用的模型深度列表')
    parser.add_argument('--more_smooth', action='store_true',
                       help='使用更平滑的生成方式（质量更好但速度更慢）')
    parser.add_argument('--save_detailed_stats', action='store_true', default=True,
                       help='保存详细的时间统计信息')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VAR (Visual Autoregressive Modeling) 图像生成推理 - 精确时间测量版本")
    print("="*70)
    
    # 检查设备可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU模式")
        args.device = 'cpu'
    elif not torch.cuda.device_count() > int(args.device.split(':')[1]):
        print(f"警告: 设备 {args.device} 不可用，切换到 cuda:0")
        args.device = 'cuda:0'
        
    # 初始化推理引擎
    engine = VARInferenceEngine(device=args.device)
    
    try:
        # 加载模型
        print("\n正在加载模型...")
        engine.load_models(args.pretrained_dir)
        
        # 设置推理环境
        engine.setup_inference_environment(args.seed)
        
        # 为每个可用的模型深度生成图像
        available_depths = [d for d in args.depths if d in engine.models]
        print(f"\n将使用以下模型深度: {available_depths}")
        
        # 存储所有模型的性能统计
        all_model_stats = []
        
        total_start_time = time.perf_counter()
        
        for depth_idx, depth in enumerate(available_depths):
            print(f"\n{'='*50}")
            print(f"开始使用VAR-d{depth}生成图像 ({depth_idx+1}/{len(available_depths)})")
            print(f"{'='*50}")
            
            # 生成图像并记录详细时间
            images, scene_names, image_stats, model_stats = engine.generate_images_with_timing(
                model_depth=depth,
                cfg_strength=args.cfg_strength,
                num_images=args.num_images,
                seed=args.seed,
                more_smooth=args.more_smooth
            )
            
            # 保存结果和统计信息
            engine.save_results_with_timing(
                images=images,
                scene_names=scene_names,
                image_stats=image_stats,
                model_stats=model_stats,
                output_dir=args.output_dir
            )
            
            all_model_stats.append(model_stats)
            
            # 清理GPU内存
            del images
            torch.cuda.empty_cache()
            
        total_time = time.perf_counter() - total_start_time
        
        # 生成总体性能比较报告
        if args.save_detailed_stats:
            comparison_report_path = Path(args.output_dir) / 'performance_comparison.json'
            comparison_data = {
                'experiment_config': {
                    'cfg_strength': args.cfg_strength,
                    'num_images_per_model': args.num_images,
                    'base_seed': args.seed,
                    'device': args.device,
                    'more_smooth': args.more_smooth
                },
                'model_comparisons': []
            }
            
            for stats in all_model_stats:
                comparison_data['model_comparisons'].append({
                    'model_depth': stats.model_depth,
                    'avg_time_per_image': round(stats.average_time_per_image, 6),
                    'total_time': round(stats.total_time, 6),
                    'throughput_images_per_second': round(stats.total_images / stats.actual_inference_time, 3),
                    'time_stability_std': round(stats.std_time, 6)
                })
            
            with open(comparison_report_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n性能比较报告已保存: {comparison_report_path}")
            
        print(f"\n{'='*70}")
        print(f"所有模型推理完成!")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"生成的模型数量: {len(available_depths)}")
        print(f"每个模型生成图像数量: {args.num_images}")
        print(f"总图像数量: {len(available_depths) * args.num_images}")
        print(f"结果保存在: {args.output_dir}")
        
        # 显示各模型性能摘要
        print(f"\n模型性能摘要:")
        print(f"{'模型深度':<10} {'平均时间(秒)':<12} {'吞吐量(图/秒)':<15} {'时间稳定性':<12}")
        print("-" * 55)
        for stats in all_model_stats:
            throughput = stats.total_images / stats.actual_inference_time
            print(f"VAR-d{stats.model_depth:<4} {stats.average_time_per_image:<12.3f} {throughput:<15.3f} ±{stats.std_time:.3f}")
            
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
