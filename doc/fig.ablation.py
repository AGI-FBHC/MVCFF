import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')


# 创建数据
methods = ['Basic','+FESNet','+SRNet', '+CVAM', '+CLM']
categories = ['MFO', 'BPO', 'CCO']

# 设置颜色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
patterns = ['/', '\\', 'x', '/', '\\']
# patterns = ['/', '\\', 'x', 'o', '.']
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
bar_width = 0.7
x = np.arange(len(methods))

# Fmax数据
fmax_data = [
    [0.412, 0.478, 0.520],
    [0.489, 0.486, 0.534],
    [0.511, 0.490, 0.541],
    [0.545, 0.511, 0.588],
    [0.597, 0.537, 0.631]
]
fmax_data = np.array(fmax_data)

plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 绘制Fmax子图
mfo_data = fmax_data[:,0]
ref_data = np.array([0.400, 0.450, 0.490, 0.500, 0.520])
bars = axes[0].bar(x,mfo_data, bar_width, color='#4e79a7',edgecolor='black', alpha=0.8, label='Our Method')
ref_bars = axes[0].bar(x,ref_data, bar_width, color="#9ea74e",edgecolor='black', alpha=0.8, label='Reference')
axes[0].bar_label(bars, labels=[f'+{v:.3f}' for v in (mfo_data-ref_data)], padding=3)
axes[0].set_xlabel('(a)', fontweight='bold')
axes[0].set_ylabel(r'$F_{max}$', fontweight='bold')
axes[0].set_ylim(0.35, 0.65)
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, rotation=25)
axes[0].set_title(categories[0], fontweight='bold')
axes[0].legend()

bpo_data = fmax_data[:,1]
bpo_bottoms = np.array([0.0, bpo_data[0], bpo_data[1], bpo_data[2], bpo_data[3]])
delta_data = bpo_data-bpo_bottoms
errors = np.array([0.001, 0.002, 0.001, 0.002, 0.001])
bar_labels =[f'+{v:.3f}' for v in (delta_data)]
bar_labels[0] = f'{bpo_data[0]:.3f}'
bars = axes[1].bar(x,delta_data, bar_width, color='#4e79a7',edgecolor='black', alpha=0.8, yerr=errors, capsize=5, bottom=bpo_bottoms)
axes[1].bar_label(bars, labels=bar_labels, padding=3)
axes[1].set_xlabel('(b)', fontweight='bold')
axes[1].set_ylabel(r'$F_{max}$', fontweight='bold')
axes[1].set_ylim(0.470, 0.550)
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, rotation=25)
axes[1].set_title(categories[1], fontweight='bold')

errors = np.array([0.001, 0.003, 0.002, 0.003, 0.004])
bars = axes[2].bar(x,fmax_data[:,2], bar_width, color='#4e79a7', edgecolor='black',alpha=0.8,yerr=errors, capsize=5)
axes[2].bar_label(bars, labels=[f'{v:.3f}' for v in fmax_data[:,2]], padding=3)
axes[2].set_xlabel('(c)', fontweight='bold')
axes[2].set_ylabel(r'$F_{max}$', fontweight='bold')
axes[2].set_ylim(0.5, 0.650)
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods, rotation=25)
axes[2].set_title(categories[2], fontweight='bold')


    
# 调整布局
plt.tight_layout()

# 保存图像
# plt.savefig('fig.performance_comparison.png', dpi=600, bbox_inches='tight')
plt.savefig('doc/fig.ablation.jpg', dpi=600, bbox_inches='tight', transparent=True)
# plt.savefig('fig.performance_comparison.pdf', bbox_inches='tight')
plt.savefig('doc/fig.ablation.svg', bbox_inches='tight', transparent=True)