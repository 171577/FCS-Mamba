import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 核心优化：确保字体可编辑的全局设置
# ==========================================
TARGET_FONT = 'Times New Roman'

try:
    plt.rcParams['font.family'] = TARGET_FONT
except:
    plt.rcParams['font.family'] = 'serif'
    
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# 强制 PDF 和 PS 嵌入 TrueType 字体，确保 AI/Inkscape 可编辑
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
# ==========================================

def extract_f1_scores(log_file, max_epochs=300):
    """提取 F1 分数"""
    f1_scores = []
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'Epoch (\d+)/\d+ \|.*val_F1_1=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                f1_score = float(match.group(2))
                if epoch <= max_epochs:
                    f1_scores.append((epoch, f1_score))
    
    f1_scores.sort(key=lambda x: x[0])
    return [score for _, score in f1_scores]

def get_checkpoint_f1_data(checkpoint_range=(171, 376), max_epochs=200):
    """获取所有 checkpoint 的 F1 数据"""
    checkpoint_data = {}
    
    for i in range(checkpoint_range[0], checkpoint_range[1] + 1):
        checkpoint_dir = f'checkpoints{i}'
        if not os.path.exists(checkpoint_dir):
            continue
        
        log_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('log_') and f.endswith('.txt')]
        if not log_files:
            continue
        
        log_file = os.path.join(checkpoint_dir, log_files[0])
        f1_scores = extract_f1_scores(log_file, max_epochs)
        
        if f1_scores and len(f1_scores) >= max_epochs:
            checkpoint_data[i] = f1_scores[:max_epochs]
    
    return checkpoint_data

def select_representative_checkpoints(checkpoint_data, num_groups=4):
    """均匀选择代表性的 checkpoint"""
    checkpoint_ids = sorted(checkpoint_data.keys())
    if len(checkpoint_ids) < num_groups:
        return checkpoint_ids
    
    indices = np.linspace(0, len(checkpoint_ids) - 1, num_groups, dtype=int)
    selected = [checkpoint_ids[i] for i in indices]
    return selected

def plot_3d_training_metrics(checkpoint_data, selected_checkpoints, max_epochs=200):
    """创建优化的 3D 可视化图表"""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    num_groups = len(selected_checkpoints)
    epochs = np.arange(1, max_epochs + 1)
    
    model_names = ['DICE+CE+0.4SIM', 'DICE+CE+0.3SIM', 'DICE+CE', 'DICE']
    bar_colors = ['#F5B7CE', '#A9DFBF', '#F9E79F', '#AED6F1'] 
    line_colors = ['#D01C8B', '#27AE60', '#D35400', '#2980B9']
    
    # 核心修改：分离柱状图和折线图的采样率
    bar_sample_rate = 10   # 柱状图每 10 个 Epoch 采样一次
    line_sample_rate = 5   # 折线图每 5 个 Epoch 采样一次（比柱子密集一倍）
    z_base = 0.7  
    
    dx = 0.25  
    dy = 2.0   

    # 第一步：先绘制所有柱状图
    for idx, checkpoint_id in enumerate(selected_checkpoints):
        if checkpoint_id not in checkpoint_data:
            continue
            
        f1_scores = checkpoint_data[checkpoint_id]
        
        # 使用柱状图采样率
        sampled_epochs = epochs[::bar_sample_rate]
        sampled_f1 = [f1_scores[i-1] for i in sampled_epochs if i <= len(f1_scores)]
        
        x_pos = np.full(len(sampled_f1), idx) 
        y_pos = sampled_epochs[:len(sampled_f1)]
        z_pos = np.full(len(sampled_f1), z_base) 
        
        dz = np.array(sampled_f1) - z_base  
        
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, 
                 color=bar_colors[idx % len(bar_colors)], alpha=0.65, edgecolor='#444444', linewidth=0.2, shade=True,
                 label=f'{model_names[idx]} Bars')

    # 第二步：再绘制所有折线图
    for idx, checkpoint_id in enumerate(selected_checkpoints):
        if checkpoint_id not in checkpoint_data:
            continue
            
        f1_scores = checkpoint_data[checkpoint_id]
        
        # 使用折线图采样率（更密集）
        sampled_epochs = epochs[::line_sample_rate]
        sampled_f1 = [f1_scores[i-1] for i in sampled_epochs if i <= len(f1_scores)]
        
        all_y = sampled_epochs[:len(sampled_f1)]
        all_z = sampled_f1
        x_line_center = idx + dx / 2
        
        ax.plot([x_line_center] * len(all_z), all_y, all_z, 
                color=line_colors[idx % len(line_colors)], linewidth=2.0, marker='o', markersize=3, 
                markerfacecolor=line_colors[idx % len(line_colors)], markeredgecolor='black', markeredgewidth=0.3,
                alpha=0.9, zorder=100, 
                label=f'{model_names[idx]} Trend')

    # ==========================================
    # 坐标轴标签与刻度优化
    # ==========================================
    ax.set_ylabel('Epoch', fontsize=14, labelpad=12)
    
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('F1 Score', fontsize=15, labelpad=12, rotation=90)
    
    ax.set_xticks([i + dx/2 for i in range(num_groups)])
    ax.set_xticklabels(model_names, fontsize=10)
    
    epoch_ticks = np.arange(0, max_epochs + 1, 50)
    ax.set_yticks(epoch_ticks)
    
    z_ticks = np.arange(0.7, 1.01, 0.05)
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'{z*100:.0f}%' for z in z_ticks])
    
    ax.set_xlim(-0.3, num_groups - 1 + dx + 0.3)
    ax.set_ylim(0, max_epochs)
    ax.set_zlim(z_base, 1.0)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=10, 
              framealpha=1.0, edgecolor='black', ncol=2)
    
    ax.view_init(elev=22, azim=-42)
    
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(1.0)
    ax.yaxis.pane.set_alpha(1.0)
    ax.zaxis.pane.set_alpha(1.0)
    ax.xaxis.pane.set_facecolor('#FFFFDD')  
    ax.yaxis.pane.set_facecolor('#FFEBD6')  
    ax.zaxis.pane.set_facecolor('#E0FFFF')  
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    
    ax.xaxis._axinfo["grid"].update({"color": "#888888", "linewidth": 0.5, "linestyle": "-"})
    ax.yaxis._axinfo["grid"].update({"color": "#888888", "linewidth": 0.5, "linestyle": "-"})
    ax.zaxis._axinfo["grid"].update({"color": "#888888", "linewidth": 0.5, "linestyle": "-"})
    
    ax.dist = 11.5 
    
    plt.subplots_adjust(left=0.05, right=0.78, top=0.82, bottom=0.1)
    
    base_name = '3d_training_metrics_final'
    plt.savefig(f'{base_name}.svg', format='svg', facecolor='white')
    plt.savefig(f'{base_name}.pdf', format='pdf', facecolor='white')
    plt.savefig(f'{base_name}.png', dpi=300, facecolor='white', bbox_inches='tight')
    
    print(f"\nFigures saved successfully! Line plot markers are now twice as dense as the bars.")
    
    plt.show()

def main():
    print("Extracting F1 scores from training logs...")
    checkpoint_data = get_checkpoint_f1_data(checkpoint_range=(171, 376), max_epochs=200)
    
    if not checkpoint_data:
        print("No checkpoint data found!")
        return
    
    selected_checkpoints = select_representative_checkpoints(checkpoint_data, num_groups=4)
    print(f"Original Base Order: {selected_checkpoints}")
    
    required_cps = [171, 246, 311, 376]
    if all(cp in selected_checkpoints for cp in required_cps):
        
        idx_246 = selected_checkpoints.index(246)
        idx_376 = selected_checkpoints.index(376)
        selected_checkpoints[idx_246], selected_checkpoints[idx_376] = selected_checkpoints[idx_376], selected_checkpoints[idx_246]
        
        idx_311 = selected_checkpoints.index(311)
        idx_246_new = selected_checkpoints.index(246)
        selected_checkpoints[idx_311], selected_checkpoints[idx_246_new] = selected_checkpoints[idx_246_new], selected_checkpoints[idx_311]
        
        print(f"Final Swapped Order: {selected_checkpoints}")
    else:
        print("Required checkpoints not fully found. Using default extracted order.")

    print("\nGenerating optimized 3D visualization with semi-dense markers...")
    plot_3d_training_metrics(checkpoint_data, selected_checkpoints, max_epochs=200)

if __name__ == '__main__':
    main()