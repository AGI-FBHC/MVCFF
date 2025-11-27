import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 18})

# 创建2行2列的子图布局，但我们将使用第一行的两个位置和第二行的第一个位置
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# 重新排列axes，使第三个图占据第二行的整个宽度
ax1 = axes[0,0]  # 第一行第一列 - Fmax
ax2 = axes[0,1]  # 第一行第二列 - Smin
ax3 = axes[0,2]  # 第一行第三列 - AUPR

ax4 = axes[1,0]  # 第二行第一列 - Fmax
ax5 = axes[1,1]  # 第二行第二列 - Smin
ax6 = axes[1,2]  # 第二行第三列 - AUPR

ax7 = axes[2,0]  # 第三行第一列 - Fmax
ax8 = axes[2,1]  # 第三行第二列 - Smin
ax9 = axes[2,2]  # 第三行第三列 - AUPR


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

bpo_smin = np.array([21.841460261196744, 21.832552261138638, 21.659883492889,
                     21.550729985327663, 21.45078954631957, 21.459860249575087,
                     21.38686607493064, 21.15985589145373, 21.250625390415043,
                     21.496339178058317, 21.450738701570384])

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
ax1.set_xlabel("(a)", fontweight='bold',fontsize=18)
ax1.set_ylabel('Fmax', fontweight='bold',fontsize=18)
ax1.set_xticks(np.arange(0, 1.1, 0.2))
ax1.set_ylim(fmax_y_min, fmax_y_max)
ax1.set_title('MFO', fontweight='bold')
ax1.scatter([x_fmax[7]], [mfo_fmax[7]], color="#ff0000", marker='o',s=100,zorder=10)
ax1.text(x_fmax[7], mfo_fmax[7]+0.01, r' $\beta$='+f'{x_fmax[7]:.1f}', fontsize=18, ha='center')


ax2.plot(x_fmax, bpo_fmax, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, label='BPO')
ax2.set_xlabel("(b)", fontweight='bold')
ax2.set_ylabel('Fmax', fontweight='bold')
ax2.set_xticks(np.arange(0, 1.1, 0.2))
ax2.set_ylim(fmax_y_min, fmax_y_max)
ax2.set_title('BPO', fontweight='bold')
ax2.scatter([x_fmax[7]], [bpo_fmax[7]], color="#ff0000", marker='o',s=100,zorder=10)
ax2.text(x_fmax[7], bpo_fmax[7]+0.01, r' $\beta$='+f'{x_fmax[7]:.1f}', fontsize=18, ha='center')

ax3.plot(x_fmax, cco_fmax, '^-', color='#2ca02c', linewidth=1.5, markersize=4, label='CCO')
ax3.set_xlabel("(c)", fontweight='bold')
ax3.set_ylabel('Fmax', fontweight='bold')
ax3.set_xticks(np.arange(0, 1.1, 0.2))
ax3.set_ylim(0.45, 0.75)
ax3.set_title('CCO', fontweight='bold')
ax3.scatter([x_fmax[9]], [cco_fmax[9]], color="#ff0000", marker='o',s=100,zorder=10)
ax3.text(x_fmax[9], cco_fmax[9]+0.01, r' $\beta$='+f'{x_fmax[9]:.1f}', fontsize=18, ha='center')

# 绘制第二个子图 - Smin
ax4.plot(x_smin, mfo_smin, 'o-', color='#1f77b4', linewidth=1.5, markersize=4, label='MPO')
ax4.set_xlabel("(d)", fontweight='bold')
ax4.set_ylabel('Smin', fontweight='bold')
ax4.set_xticks(np.arange(0, 1.1, 0.2))
ax4.set_ylim(smin_y_min, 12)
ax4.set_title('MFO', fontweight='bold')
ax4.scatter([x_smin[6]], [mfo_smin[6]], color="#ff0000", marker='o',s=100,zorder=10)
ax4.text(x_smin[6], mfo_smin[6]-0.7, r' $\beta$='+f'{x_smin[6]:.1f}', fontsize=18, ha='center')

ax5.plot(x_smin, bpo_smin, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, label='BPO')
ax5.set_xlabel("(e)", fontweight='bold')
ax5.set_ylabel('Smin', fontweight='bold')
ax5.set_xticks(np.arange(0, 1.1, 0.2))
ax5.set_ylim(20, 25)
ax5.set_title('BPO', fontweight='bold')
ax5.scatter([x_smin[7]], [bpo_smin[7]], color="#ff0000", marker='o',s=100,zorder=10)
ax5.text(x_smin[7]-0.1, bpo_smin[7]-0.6 , r' $\beta$='+f'{x_smin[7]:.1f}', fontsize=18, ha='center')

ax6.plot(x_smin, cco_smin, '^-', color='#2ca02c', linewidth=1.5, markersize=4, label='CCO')
ax6.set_xlabel("(f)", fontweight='bold')
ax6.set_ylabel('Smin', fontweight='bold')
ax6.set_xticks(np.arange(0, 1.1, 0.2))
ax6.set_ylim(7, 9)
ax6.set_title('CCO', fontweight='bold')
ax6.scatter([x_smin[9]], [cco_smin[9]], color="#ff0000", marker='o',s=100,zorder=10)
ax6.text(x_smin[9], cco_smin[9]-0.2, r' $\beta$='+f'{x_smin[9]:.1f}', fontsize=18, ha='center')

# 绘制第三个子图 - AUPR
ax7.plot(x_aupr, mfo_aupr, 'o-', color='#1f77b4', linewidth=1.5, markersize=4, label='MPO')
ax7.set_xlabel("(g)", fontweight='bold')
ax7.set_ylabel('AUPR', fontweight='bold')
ax7.set_xticks(np.arange(0, 1.1, 0.2))
ax7.set_ylim(aupr_y_min, aupr_y_max)
ax7.set_title('MFO', fontweight='bold')
ax7.scatter([x_aupr[7]], [mfo_aupr[7]], color="#ff0000", marker='o',s=100,zorder=10)
ax7.text(x_aupr[7], mfo_aupr[7]+0.01, r' $\beta$='+f'{x_aupr[7]:.1f}', fontsize=18, ha='center')

ax8.plot(x_aupr, bpo_aupr, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, label='BPO')
ax8.set_xlabel("(h)", fontweight='bold')
ax8.set_ylabel('AUPR', fontweight='bold')
ax8.set_xticks(np.arange(0, 1.1, 0.2))
ax8.set_ylim(aupr_y_min, aupr_y_max)
ax8.set_title('BPO', fontweight='bold')
ax8.scatter([x_aupr[7]], [bpo_aupr[7]], color="#ff0000", marker='o',s=100,zorder=10)
ax8.text(x_aupr[7], bpo_aupr[7]+0.01, r' $\beta$='+f'{x_aupr[7]:.1f}', fontsize=18, ha='center')

ax9.plot(x_aupr, cco_aupr, '^-', color='#2ca02c', linewidth=1.5, markersize=4, label='CCO')
ax9.set_xlabel("(i)", fontweight='bold')
ax9.set_ylabel('AUPR', fontweight='bold')
ax9.set_xticks(np.arange(0, 1.1, 0.2))
ax9.set_ylim(aupr_y_min, 0.8)
ax9.set_title('CCO', fontweight='bold')
ax9.scatter([x_aupr[9]], [cco_aupr[9]], color="#ff0000", marker='o',s=100,zorder=10)  
ax9.text(x_aupr[9], cco_aupr[9]+0.03, r' $\beta$='+f'{x_aupr[9]:.1f}', fontsize=18, ha='center')

# 调整布局，确保子图之间有适当的间距
plt.tight_layout()
plt.savefig('doc/fig.balanceablation.jpg', dpi=600, bbox_inches='tight',transparent=True)  # 高DPI版本
plt.savefig('doc/fig.balanceablation.svg', bbox_inches='tight', transparent=True)  # 矢量图版本

plt.show()