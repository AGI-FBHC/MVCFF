import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和DPI
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# 创建2行2列的子图布局，但我们将使用第一行的两个位置和第二行的第一个位置
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# 重新排列axes，使第三个图占据第二行的整个宽度
ax1 = axes[0, 0]  # 第一行第一列 - Fmax
ax2 = axes[0, 1]  # 第一行第二列 - Smin
ax3 = axes[1, 0]  # 第二行第一列 - AUPR

# 隐藏第二行第二列的空子图
axes[1, 1].set_visible(False)

# Fmax 数据
x_fmax = np.array([0, 0.1, 0.2, 0.3,
                   0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9, 1])

mfo_fmax = np.array([0.5090082455336693, 0.5298568483737975, 0.5470190105359597,
                     0.5532420980302337, 0.5585913879981677, 0.5599816765918462,
                     0.5654535043518094, 0.5707913421896473, 0.5627782867613377,
                     0.552241181859826, 0.5493151626202474])

bpo_fmax = np.array([0.4275217590471829, 0.47442739349519014, 0.48625629867155296,
                     0.5036704076958315, 0.5242659184608338, 0.5345590929912963,
                     0.5412998167659185, 0.5555382501145214, 0.5406665139715988,
                     0.5318964727439304, 0.520328676133761])

cco_fmax = np.array([0.5575687127805773, 0.5711601007787449, 0.5719388456252863,
                     0.5730817682088869, 0.5768976179569401, 0.5916342189647275,
                     0.6057478240952817, 0.6229134218964728, 0.6221495648190564,
                     0.6399427393495191, 0.6343540998625744])

# Smin 数据
x_smin = np.array([0, 0.1, 0.2, 0.3,
                   0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9, 1])

mfo_smin = np.array([9.00509900199021, 8.695861237415926, 8.541333875677324,
                     8.450403126225726, 8.405130961546849, 8.323169225852382,
                     8.150500457602746, 8.159652512456969, 8.16860118831443,
                     8.223411827941376, 8.323270915350765])

bpo_smin = np.array([22.841460261196744, 22.832552261138638, 22.659883492889,
                     22.550729985327663, 22.45078954631957, 22.459860249575087,
                     22.28686607493064, 22.15985589145373, 22.350625390415043,
                     21.996339178058317, 21.750738701570384])

cco_smin = np.array([8.232157124802075, 8.104943562328401, 8.114095617182624,
                     8.122840914043323, 8.086944521115104, 8.050437991196599,
                     7.968679634498894, 7.841669451021982, 7.8504147478826845,
                     7.7686563911849795, 7.823263651815164])

# AUPR 数据
x_aupr = np.array([0, 0.1, 0.2, 0.3,
                   0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9, 1])

mfo_aupr = np.array([0.3419305734694278, 0.45204508403333826, 0.48448830342353194,
                     0.510418889987364, 0.5289060934809509, 0.5290223717894158,
                     0.5337897824364758, 0.5353019242226835, 0.5212312013461047,
                     0.5167005375252887, 0.5098380222212038])

bpo_aupr = np.array([0.26634810163876455, 0.424368227737876, 0.44565134839626275,
                     0.45111485756561936, 0.45169729666027214, 0.45414071246652743,
                     0.46309152333750175, 0.47251163765164966, 0.4703013022384883,
                     0.47018659525851625, 0.47007031695005136])

cco_aupr = np.array([0.3335580114837924, 0.5292502144208673, 0.5479762598453551,
                     0.559718797671815, 0.5721626717822139, 0.5832054446532275,
                     0.5951852530820954, 0.606001230873986, 0.6307658916961837,
                     0.6532107478868905, 0.6368144588410143])

# 计算每个子图的扩展范围
# Fmax: y轴范围从0.35-0.65扩展到0.33-0.67
fmax_y_min, fmax_y_max = 0.37, 0.67

# Smin: y轴范围从7-23.5扩展到6-24.5
smin_y_min, smin_y_max = 6.5, 23.5

# AUPR: y轴范围从0.25-0.65扩展到0.23-0.67
aupr_y_min, aupr_y_max = 0.23, 0.67

# 绘制第一个子图 - Fmax
ax1.plot(x_fmax, mfo_fmax, 'o-', color='#1f77b4', linewidth=1.5, markersize=4, label='MPO')
ax1.plot(x_fmax, bpo_fmax, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, label='BPO')
ax1.plot(x_fmax, cco_fmax, '^-', color='#2ca02c', linewidth=1.5, markersize=4, label='CCO')
ax1.set_xlim(-0.05, 1.05)  # x轴稍微扩展
ax1.set_ylim(fmax_y_min, fmax_y_max)  # y轴扩展
ax1.set_ylabel('Scores', fontweight='bold')
# 将标题放在子图上方
ax1.text(0.5, 1.05, 'Fmax', transform=ax1.transAxes, fontsize=12, fontweight='bold', ha='center')
ax1.text(0.02, 0.95, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

# 绘制第二个子图 - Smin
ax2.plot(x_smin, mfo_smin, 'o-', color='#1f77b4', linewidth=1.5, markersize=4, label='MPO')
ax2.plot(x_smin, bpo_smin, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, label='BPO')
ax2.plot(x_smin, cco_smin, '^-', color='#2ca02c', linewidth=1.5, markersize=4, label='CCO')
ax2.set_xlim(-0.05, 1.05)  # x轴稍微扩展
ax2.set_ylim(smin_y_min, smin_y_max)  # y轴扩展
ax2.set_ylabel('Scores', fontweight='bold')
# 将标题放在子图上方
ax2.text(0.5, 1.05, 'Smin', transform=ax2.transAxes, fontsize=12, fontweight='bold', ha='center')
ax2.text(0.02, 0.95, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
ax2.grid(True, linestyle='--', alpha=0.7)
# 将图例放在中间且不遮挡线条
ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.5), frameon=True, fancybox=True, shadow=True)

# 绘制第三个子图 - AUPR
ax3.plot(x_aupr, mfo_aupr, 'o-', color='#1f77b4', linewidth=1.5, markersize=4, label='MPO')
ax3.plot(x_aupr, bpo_aupr, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, label='BPO')
ax3.plot(x_aupr, cco_aupr, '^-', color='#2ca02c', linewidth=1.5, markersize=4, label='CCO')
ax3.set_xlim(-0.05, 1.05)  # x轴稍微扩展
ax3.set_ylim(aupr_y_min, aupr_y_max)  # y轴扩展
ax3.set_ylabel('Scores', fontweight='bold')
# 将标题放在子图上方
ax3.text(0.5, 1.05, 'AUPR', transform=ax3.transAxes, fontsize=12, fontweight='bold', ha='center')
ax3.text(0.02, 0.95, 'C', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

# 调整布局，确保子图之间有适当的间距
plt.tight_layout()

# 保存高DPI图像
plt.savefig('high_dpi_plot.png', dpi=600, bbox_inches='tight')
plt.savefig('high_dpi_plot.pdf', bbox_inches='tight')  # 矢量图版本

plt.show()