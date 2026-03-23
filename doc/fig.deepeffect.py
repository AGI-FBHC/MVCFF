import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')


# 创建数据
methods = ['Basic','+FESNet','+SRNet', '+CVAM', '+CLM']
categories = ['MFO', 'BPO', 'CCO']

# Fmax数据
fmax_data = [
    [0.512, 0.478, 0.601],
    [0.536, 0.496, 0.622],
    [0.541, 0.503, 0.626],
    [0.545, 0.511, 0.628],
    [0.547, 0.517, 0.631]
]
fmax_data = np.array(fmax_data)
  

# Smin数据
smin_data = [
    [9.123, 23.456, 8.234], 
    [8.789, 22.987, 7.945],
    [8.512, 22.543, 7.856],
    [8.423, 22.123, 7.789],
    [8.274, 21.683, 7.704]
]
smin_data = np.array(smin_data)

# AUPR数据
aupr_data = [
    [0.482, 0.451, 0.612],
    [0.501, 0.463, 0.623],
    [0.506, 0.467, 0.628],
    [0.506, 0.467, 0.630],
    [0.508, 0.469, 0.634]
]
aupr_data = np.array(aupr_data)
# 创建图形和子图
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# 设置颜色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
patterns = ['/', '\\', 'x', '/', '\\']
# patterns = ['/', '\\', 'x', 'o', '.']
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
bar_width = 0.7
x = np.arange(len(methods))

# 绘制Fmax子图
bars = axes[0,0].bar(x,fmax_data[:,0], bar_width, color=colors,edgecolor='black', alpha=0.8)
axes[0,0].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,0]], padding=3)
axes[0,0].set_xlabel('(a)', fontweight='bold')
axes[0,0].set_ylabel(r'$F_{max}$', fontweight='bold')
axes[0,0].set_ylim(0.5, 0.58)
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(methods, rotation=25)
axes[0,0].set_title(categories[0], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

bars = axes[0,1].bar(x,fmax_data[:,1], bar_width, color=colors, alpha=0.8)
axes[0,1].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,1]], padding=3)
axes[0,1].set_xlabel('(b)', fontweight='bold')
axes[0,1].set_ylabel(r'$F_{max}$', fontweight='bold')
axes[0,1].set_ylim(0.470, 0.550)
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(methods, rotation=25)
axes[0,1].set_title(categories[1], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

bars = axes[0,2].bar(x,fmax_data[:,2], bar_width, color=colors, alpha=0.8)
axes[0,2].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,2]], padding=3)
axes[0,2].set_xlabel('(c)', fontweight='bold')
axes[0,2].set_ylabel(r'$F_{max}$', fontweight='bold')
axes[0,2].set_ylim(0.6, 0.640)
axes[0,2].set_xticks(x)
axes[0,2].set_xticklabels(methods, rotation=25)
axes[0,2].set_title(categories[2], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

# 绘制Smin子图
bars = axes[1,0].bar(x,smin_data[:,0], bar_width, color=colors, alpha=0.8)
axes[1,0].bar_label(bars, labels=[f'{v:.3f}' for v in smin_data[:,0]], padding=3)
axes[1,0].set_xlabel('(d)', fontweight='bold')
axes[1,0].set_ylabel(r'$S_{min}$', fontweight='bold')
axes[1,0].set_ylim(8, 9.5)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(methods, rotation=25)
axes[1,0].set_title(categories[0], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

bars = axes[1,1].bar(x,smin_data[:,1], bar_width, color=colors, alpha=0.8)
axes[1,1].bar_label(bars, labels=[f'{v:.3f}' for v in smin_data[:,1]], padding=3)
axes[1,1].set_xlabel('(e)', fontweight='bold')
axes[1,1].set_ylabel(r'$S_{min}$', fontweight='bold')
axes[1,1].set_ylim(21, 24)
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(methods, rotation=25)
axes[1,1].set_title(categories[1], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

bars = axes[1,2].bar(x,smin_data[:,2], bar_width, color=colors, alpha=0.8)
axes[1,2].bar_label(bars, labels=[f'{v:.3f}' for v in smin_data[:,2]], padding=3)
axes[1,2].set_xlabel('(f)', fontweight='bold')
axes[1,2].set_ylabel(r'$S_{min}$', fontweight='bold')
axes[1,2].set_ylim(7.5, 8.5)
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(methods, rotation=25)
axes[1,2].set_title(categories[2], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

# 绘制AUPR子图
bars = axes[2,0].bar(x,fmax_data[:,0], bar_width, color=colors, alpha=0.8)
axes[2,0].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,0]], padding=3)
axes[2,0].set_xlabel('(g)', fontweight='bold')
axes[2,0].set_ylabel('AUPR', fontweight='bold')
axes[2,0].set_ylim(0.5, 0.58)
axes[2,0].set_xticks(x)
axes[2,0].set_xticklabels(methods, rotation=25)
axes[2,0].set_title(categories[0], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

bars = axes[2,1].bar(x,fmax_data[:,1], bar_width, color=colors, alpha=0.8)
axes[2,1].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,1]], padding=3)
axes[2,1].set_xlabel('(h)', fontweight='bold')
axes[2,1].set_ylabel('AUPR', fontweight='bold')
axes[2,1].set_ylim(0.470, 0.550)
axes[2,1].set_xticks(x)
axes[2,1].set_xticklabels(methods, rotation=25)
axes[2,1].set_title(categories[1], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

bars = axes[2,2].bar(x,fmax_data[:,2], bar_width, color=colors, alpha=0.8)
axes[2,2].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,2]], padding=3)
axes[2,2].set_xlabel('(i)', fontweight='bold')
axes[2,2].set_ylabel('AUPR', fontweight='bold')
axes[2,2].set_ylim(0.6, 0.640)
axes[2,2].set_xticks(x)
axes[2,2].set_xticklabels(methods, rotation=25)
axes[2,2].set_title(categories[2], fontweight='bold')
# 为每个柱添加不同的纹理
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)


# 调整布局
plt.tight_layout()

# 保存图像
# plt.savefig('fig.performance_comparison.png', dpi=600, bbox_inches='tight')
plt.savefig('doc/fig.deepeffect.jpg', dpi=600, bbox_inches='tight', transparent=True)
# plt.savefig('fig.performance_comparison.pdf', bbox_inches='tight')
plt.savefig('doc/fig.deepeffect.svg', bbox_inches='tight', transparent=True)

# 显示图形
plt.show()