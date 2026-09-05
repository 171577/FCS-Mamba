"""
生成弱标签的工具脚本

用法:
    python generate_weak_labels.py --dataroot /path/to/dataset --method dilate --kernel_size 15
    python generate_weak_labels.py --dataroot /path/to/dataset --method downsample --scale 8
    python generate_weak_labels.py --dataroot /path/to/dataset --method image_level
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def generate_weak_label_dilate(strong_label, kernel_size=15):
    """
    通过膨胀操作生成弱标签
    
    Args:
        strong_label: 强标签图像 (0/255)
        kernel_size: 膨胀核大小
    
    Returns:
        弱标签图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    weak_label = cv2.dilate(strong_label, kernel, iterations=1)
    return weak_label


def generate_weak_label_downsample(strong_label, scale=8):
    """
    通过下采样+上采样生成弱标签
    
    Args:
        strong_label: 强标签图像 (0/255)
        scale: 下采样倍数
    
    Returns:
        弱标签图像
    """
    h, w = strong_label.shape
    small = cv2.resize(strong_label, (w//scale, h//scale), 
                       interpolation=cv2.INTER_NEAREST)
    weak_label = cv2.resize(small, (w, h), 
                           interpolation=cv2.INTER_NEAREST)
    return weak_label


def generate_weak_label_image_level(strong_label):
    """
    生成图像级弱标签（整张图有/无变化）
    
    Args:
        strong_label: 强标签图像 (0/255)
    
    Returns:
        弱标签图像
    """
    has_change = np.any(strong_label > 0)
    if has_change:
        weak_label = np.ones_like(strong_label) * 255
    else:
        weak_label = np.zeros_like(strong_label)
    return weak_label


def generate_weak_label_erode_dilate(strong_label, erode_size=3, dilate_size=15):
    """
    先腐蚀后膨胀，生成更粗糙的弱标签
    
    Args:
        strong_label: 强标签图像 (0/255)
        erode_size: 腐蚀核大小
        dilate_size: 膨胀核大小
    
    Returns:
        弱标签图像
    """
    # 先腐蚀，去除小的噪声
    erode_kernel = np.ones((erode_size, erode_size), np.uint8)
    eroded = cv2.erode(strong_label, erode_kernel, iterations=1)
    
    # 再膨胀，扩大区域
    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    weak_label = cv2.dilate(eroded, dilate_kernel, iterations=1)
    
    return weak_label


def process_dataset(dataroot, method='dilate', phases=['train', 'val', 'test'], **kwargs):
    """
    处理整个数据集，生成弱标签
    
    Args:
        dataroot: 数据集根目录
        method: 生成方法 ('dilate', 'downsample', 'image_level', 'erode_dilate')
        phases: 要处理的阶段列表
        **kwargs: 方法特定的参数
    """
    dataroot = Path(dataroot)
    
    # 方法映射
    method_map = {
        'dilate': generate_weak_label_dilate,
        'downsample': generate_weak_label_downsample,
        'image_level': generate_weak_label_image_level,
        'erode_dilate': generate_weak_label_erode_dilate,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
    
    generate_func = method_map[method]
    
    print(f"Generating weak labels using method: {method}")
    print(f"Parameters: {kwargs}")
    print(f"Phases: {phases}")
    print("-" * 80)
    
    total_processed = 0
    total_skipped = 0
    
    for phase in phases:
        label_dir = dataroot / phase / 'label'
        label_weak_dir = dataroot / phase / 'label_weak'
        
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist, skipping {phase}")
            continue
        
        # 创建弱标签目录
        label_weak_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有标签文件
        label_files = sorted(label_dir.glob('*.png')) + sorted(label_dir.glob('*.jpg'))
        
        if len(label_files) == 0:
            print(f"Warning: No label files found in {label_dir}")
            continue
        
        print(f"\nProcessing {phase} phase: {len(label_files)} images")
        
        for label_path in tqdm(label_files, desc=f"{phase}"):
            # 读取强标签
            strong_label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            if strong_label is None:
                print(f"Warning: Failed to read {label_path}, skipping")
                total_skipped += 1
                continue
            
            # 生成弱标签
            try:
                weak_label = generate_func(strong_label, **kwargs)
            except Exception as e:
                print(f"Error processing {label_path}: {e}")
                total_skipped += 1
                continue
            
            # 保存弱标签
            weak_label_path = label_weak_dir / label_path.name
            cv2.imwrite(str(weak_label_path), weak_label)
            
            total_processed += 1
    
    print("\n" + "=" * 80)
    print(f"Generation complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print("=" * 80)


def visualize_comparison(dataroot, phase='train', num_samples=5, method='dilate', **kwargs):
    """
    可视化强标签和弱标签的对比
    
    Args:
        dataroot: 数据集根目录
        phase: 阶段
        num_samples: 可视化样本数
        method: 生成方法
        **kwargs: 方法参数
    """
    import matplotlib.pyplot as plt
    
    dataroot = Path(dataroot)
    label_dir = dataroot / phase / 'label'
    
    if not label_dir.exists():
        print(f"Error: {label_dir} does not exist")
        return
    
    label_files = sorted(label_dir.glob('*.png'))[:num_samples]
    
    method_map = {
        'dilate': generate_weak_label_dilate,
        'downsample': generate_weak_label_downsample,
        'image_level': generate_weak_label_image_level,
        'erode_dilate': generate_weak_label_erode_dilate,
    }
    
    generate_func = method_map[method]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, label_path in enumerate(label_files):
        strong_label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        weak_label = generate_func(strong_label, **kwargs)
        
        axes[i, 0].imshow(strong_label, cmap='gray')
        axes[i, 0].set_title(f'Strong Label: {label_path.name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(weak_label, cmap='gray')
        axes[i, 1].set_title(f'Weak Label (method={method})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'weak_label_comparison_{method}.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: weak_label_comparison_{method}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate weak labels from strong labels')
    
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--method', type=str, default='dilate',
                        choices=['dilate', 'downsample', 'image_level', 'erode_dilate'],
                        help='Method to generate weak labels')
    parser.add_argument('--phases', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Phases to process')
    
    # Method-specific parameters
    parser.add_argument('--kernel_size', type=int, default=15,
                        help='Kernel size for dilate method')
    parser.add_argument('--scale', type=int, default=8,
                        help='Downsampling scale for downsample method')
    parser.add_argument('--erode_size', type=int, default=3,
                        help='Erosion kernel size for erode_dilate method')
    parser.add_argument('--dilate_size', type=int, default=15,
                        help='Dilation kernel size for erode_dilate method')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize comparison before generating')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Prepare method-specific kwargs
    kwargs = {}
    if args.method == 'dilate':
        kwargs['kernel_size'] = args.kernel_size
    elif args.method == 'downsample':
        kwargs['scale'] = args.scale
    elif args.method == 'erode_dilate':
        kwargs['erode_size'] = args.erode_size
        kwargs['dilate_size'] = args.dilate_size
    
    # Visualize if requested
    if args.visualize:
        print("Generating visualization...")
        visualize_comparison(
            args.dataroot,
            phase=args.phases[0],
            num_samples=args.num_samples,
            method=args.method,
            **kwargs
        )
        
        response = input("\nDo you want to proceed with generation? (y/n): ")
        if response.lower() != 'y':
            print("Generation cancelled.")
            return
    
    # Generate weak labels
    process_dataset(
        args.dataroot,
        method=args.method,
        phases=args.phases,
        **kwargs
    )


if __name__ == '__main__':
    main()
