import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持（如果需要显示中文）
plt.rcParams['font.family'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
methods = ['DFEM', 'DFEM + MVFCLN']
categories = ['MFO', 'BPO', 'CCO']

# Fmax数据
fmax_data = {
    'DFEM': [0.536, 0.496, 0.622],
    'DFEM + MVFCLN': [0.547, 0.517, 0.631]
}

# Smin数据
smin_data = {
    'DFEM': [8.366, 22.376, 7.735],
    'DFEM + MVFCLN': [8.274, 21.683, 7.704]
}

# AUPR数据
aupr_data = {
    'DFEM': [0.495, 0.462, 0.626],
    'DFEM + MVFCLN': [0.508, 0.469, 0.634]
}

# 创建图形和子图
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# 设置颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bar_width = 0.35
x = np.arange(len(categories))

# 绘制Fmax子图
for i, method in enumerate(methods):
    offset = bar_width * i
    axes[0].bar(x + offset, fmax_data[method], bar_width,
                label=method, color=colors[i], alpha=0.8)

axes[0].set_xlabel('Ontology', fontweight='bold')
axes[0].set_ylabel('Score', fontweight='bold')
axes[0].set_title('Fmax', fontweight='bold')
axes[0].set_xticks(x + bar_width / 2)
axes[0].set_xticklabels(categories)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# 绘制Smin子图
for i, method in enumerate(methods):
    offset = bar_width * i
    axes[1].bar(x + offset, smin_data[method], bar_width,
                label=method, color=colors[i], alpha=0.8)

axes[1].set_xlabel('Ontology', fontweight='bold')
axes[1].set_ylabel('Score', fontweight='bold')
axes[1].set_title('Smin', fontweight='bold')
axes[1].set_xticks(x + bar_width / 2)
axes[1].set_xticklabels(categories)
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# 绘制AUPR子图
for i, method in enumerate(methods):
    offset = bar_width * i
    axes[2].bar(x + offset, aupr_data[method], bar_width,
                label=method, color=colors[i], alpha=0.8)

axes[2].set_xlabel('Ontology', fontweight='bold')
axes[2].set_ylabel('Score', fontweight='bold')
axes[2].set_title('AUPR', fontweight='bold')
axes[2].set_xticks(x + bar_width / 2)
axes[2].set_xticklabels(categories)
axes[2].legend()
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('performance_comparison.pdf', bbox_inches='tight')

# 显示图形
plt.show()