import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(3, 3, figsize=(12, 12))



alpha = np.array([0.1,0.01,0.001,0.0001,0.00001])

mfo_fmax = np.array([0.4890082455336693, 0.5298568483737975, 0.5170190105359597,
                     0.5132420980302337, 0.5185913879981677])

bpo_fmax = np.array([0.5275217590471829, 0.54442739349519014, 0.53625629867155296,
                     0.5036704076958315, 0.5242659184608338])

cco_fmax = np.array([0.5575687127805773, 0.5811601007787449, 0.5719388456252863,
                     0.5730817682088869, 0.5768976179569401])

mfo_smin = np.array([9.00509900199021, 8.695861237415926, 8.941333875677324,
                     8.950403126225726, 8.905130961546849])

bpo_smin = np.array([23.841460261196744, 21.832552261138638, 22.659883492889,
                     22.550729985327663, 22.45078954631957])

cco_smin = np.array([8.432157124802075, 8.004943562328401, 8.114095617182624,
                     8.122840914043323, 8.186944521115104])

mfo_aupr = np.array([0.3419305734694278, 0.53204508403333826, 0.48448830342353194,
                     0.510418889987364, 0.5289060934809509])

bpo_aupr = np.array([0.36634810163876455, 0.464368227737876, 0.44565134839626275,
                     0.45111485756561936, 0.45169729666027214])

cco_aupr = np.array([0.3335580114837924, 0.5592502144208673, 0.5479762598453551,
                     0.559718797671815, 0.5421626717822139])


plt.subplot(3,3,1)
plt.bar(range(len(alpha)), mfo_fmax, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), mfo_fmax, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [mfo_fmax[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(a)', fontweight='bold')
plt.ylim(0.48, 0.58)
plt.ylabel(r'$F_{max}$', fontweight='bold')
plt.title('MFO', fontweight='bold')
plt.text(1, mfo_fmax[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,2)
plt.bar(range(len(alpha)), bpo_fmax, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), bpo_fmax, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [bpo_fmax[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(b)', fontweight='bold')
plt.ylim(0.48, 0.60)
plt.ylabel(r'$F_{max}$', fontweight='bold')
plt.title('BPO', fontweight='bold')
plt.text(1, bpo_fmax[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,3)
plt.bar(range(len(alpha)), cco_fmax, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), cco_fmax, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [cco_fmax[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(c)', fontweight='bold')
plt.ylim(0.48, 0.60)
plt.ylabel(r'$F_{max}$', fontweight='bold')
plt.title('CCO', fontweight='bold')
plt.text(1, cco_fmax[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,4)
plt.bar(range(len(alpha)), mfo_smin, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), mfo_smin, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [mfo_smin[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(d)', fontweight='bold')
plt.ylim(8.3, 9.2)
plt.ylabel(r'$S_{min}$', fontweight='bold')
plt.title('MFO', fontweight='bold')
plt.text(1, mfo_smin[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,5)
plt.bar(range(len(alpha)), bpo_smin, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), bpo_smin, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [bpo_smin[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(e)', fontweight='bold')
plt.ylim(21.4, 24.0)
plt.ylabel(r'$S_{min}$', fontweight='bold')
plt.title('BPO', fontweight='bold')
plt.text(1, bpo_smin[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,6)
plt.bar(range(len(alpha)), cco_smin, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), cco_smin, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [cco_smin[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(f)', fontweight='bold')
plt.ylim(7.8, 8.7)
plt.ylabel(r'$S_{min}$', fontweight='bold')
plt.title('CCO', fontweight='bold')
plt.text(1, cco_smin[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,7)
plt.bar(range(len(alpha)), mfo_aupr, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), mfo_aupr, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [mfo_aupr[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(g)', fontweight='bold')
plt.ylim(0.3, 0.6)
plt.ylabel('AUPR', fontweight='bold')
plt.title('MFO', fontweight='bold')
plt.text(1, mfo_aupr[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,8)
plt.bar(range(len(alpha)), bpo_aupr, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), bpo_aupr, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [bpo_aupr[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(h)', fontweight='bold')
plt.ylim(0.3, 0.6)
plt.ylabel('AUPR', fontweight='bold')
plt.title('BPO', fontweight='bold')
plt.text(1, bpo_aupr[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')

plt.subplot(3,3,9)
plt.bar(range(len(alpha)), cco_aupr, width=0.5, color="#1faab4", edgecolor='black')
plt.plot(range(len(alpha)), cco_aupr, 'o-', color="#b43d1f", linewidth=1.5, markersize=4)
plt.scatter([1], [cco_aupr[1]], color="#ff0000", marker='o',s=100,zorder=10)
plt.xticks(np.arange(0, len(alpha), 1), alpha, rotation=30)
plt.xlabel('(i)', fontweight='bold')
plt.ylim(0.3, 0.6)
plt.ylabel('AUPR', fontweight='bold')
plt.title('CCO', fontweight='bold')
plt.text(1, cco_aupr[1]+0.005, r' $\alpha$='+f'{alpha[1]:.2f}', fontsize=18, ha='center')


# 调整布局，确保子图之间有适当的间距
plt.tight_layout()
plt.savefig('doc/fig.l2.jpg', dpi=600, bbox_inches='tight',transparent=True)  # 高DPI版本
plt.savefig('doc/fig.l2.svg', bbox_inches='tight', transparent=True)  # 矢量图版本

plt.show()