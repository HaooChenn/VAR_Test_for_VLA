#!/usr/bin/env python3
"""
VAR (Visual Autoregressive Modeling) 推理脚本
支持多种深度模型，生成高质量图像
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torchvision
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw

# 禁用默认参数初始化以加速
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var

class VARInferenceEngine:
    """VAR推理引擎，支持多种模型深度和生成配置"""
    
    def __init__(self, device: str = 'cuda:1'):
        self.device = device
        self.models = {}  # 存储不同深度的模型
        self.vae = None
        
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
        
    def generate_images(self, 
                       model_depth: int,
                       cfg_strength: float = 3.0,
                       num_images: int = 20,
                       seed: int = 42,
                       more_smooth: bool = False) -> Tuple[torch.Tensor, List[str]]:
        """
        使用指定深度的模型生成图像
        
        Args:
            model_depth: 模型深度 (16, 20, 24, 30)
            cfg_strength: Classifier-free guidance强度
            num_images: 生成图像数量
            seed: 随机种子
            more_smooth: 是否使用更平滑的生成方式
            
        Returns:
            生成的图像张量和对应的类别名称列表
        """
        if model_depth not in self.models:
            raise ValueError(f"模型深度 {model_depth} 不可用，可用深度: {list(self.models.keys())}")
            
        var_model = self.models[model_depth]
        
        # 选择要生成的类别（循环使用确保有20张图）
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
        
        # 转换为张量
        label_tensor = torch.tensor(class_labels, device=self.device)
        
        # 开始生成
        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                generated_images = var_model.autoregressive_infer_cfg(
                    B=num_images,
                    label_B=label_tensor,
                    cfg=cfg_strength,
                    top_k=900,
                    top_p=0.95,
                    g_seed=seed,
                    more_smooth=more_smooth
                )
        
        generation_time = time.time() - start_time
        print(f"生成完成 - 耗时: {generation_time:.2f}秒 ({generation_time/num_images:.2f}秒/张)")
        
        return generated_images, selected_scenes
        
    def save_results(self, 
                    images: torch.Tensor,
                    scene_names: List[str],
                    model_depth: int,
                    output_dir: str = './outputs',
                    cfg_strength: float = 3.0) -> None:
        """保存生成结果"""
        output_path = Path(output_dir)
        model_output_dir = output_path / f'var_d{model_depth}_cfg{cfg_strength}'
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存单独的图像
        for i, (img, scene_name) in enumerate(zip(images, scene_names)):
            img_pil = torchvision.transforms.ToPILImage()(img)
            img_pil.save(model_output_dir / f'{i:02d}_{scene_name}.png')
            
        # 创建网格图像
        grid_img = torchvision.utils.make_grid(
            images, 
            nrow=5,  # 5列
            padding=2, 
            pad_value=1.0
        )
        grid_pil = torchvision.transforms.ToPILImage()(grid_img)
        grid_pil.save(model_output_dir / f'grid_var_d{model_depth}.png')
        
        print(f"结果已保存至: {model_output_dir}")
        print(f"- 单独图像: {len(images)} 张")
        print(f"- 网格图像: grid_var_d{model_depth}.png")

def main():
    parser = argparse.ArgumentParser(description='VAR图像生成推理脚本')
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
                       help='随机种子')
    parser.add_argument('--depths', nargs='+', type=int, default=[16, 20, 24, 30],
                       help='要使用的模型深度列表')
    parser.add_argument('--more_smooth', action='store_true',
                       help='使用更平滑的生成方式（质量更好但速度更慢）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VAR (Visual Autoregressive Modeling) 图像生成推理")
    print("="*60)
    
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
        
        total_start_time = time.time()
        
        for depth in available_depths:
            print(f"\n{'='*40}")
            print(f"开始使用VAR-d{depth}生成图像")
            print(f"{'='*40}")
            
            # 生成图像
            images, scene_names = engine.generate_images(
                model_depth=depth,
                cfg_strength=args.cfg_strength,
                num_images=args.num_images,
                seed=args.seed,
                more_smooth=args.more_smooth
            )
            
            # 保存结果
            engine.save_results(
                images=images,
                scene_names=scene_names,
                model_depth=depth,
                output_dir=args.output_dir,
                cfg_strength=args.cfg_strength
            )
            
            # 清理GPU内存
            del images
            torch.cuda.empty_cache()
            
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"所有模型推理完成!")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"生成的模型数量: {len(available_depths)}")
        print(f"每个模型生成图像数量: {args.num_images}")
        print(f"总图像数量: {len(available_depths) * args.num_images}")
        print(f"结果保存在: {args.output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
