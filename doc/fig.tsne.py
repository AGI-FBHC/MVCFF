import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')
# plt.rcParams.update({'font.size': 18})

data_all = np.load('doc/fig.tsne.data.npy')
labels = np.load('doc/fig.tsne.labels.npy')
views = ["One-hot", "OPF", "PSSM", "ESM"]

# t-SNE降维到2维
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_all)

# 可视化
plt.figure(figsize=(5,5))
for i in range(4):
    plt.scatter(data_tsne[labels==i, 0], data_tsne[labels==i, 1], label=views[i], alpha=0.4, s=50)

plt.title('t-SNE Visualization', fontweight='bold')
plt.xlabel('Dimension 1', fontweight='bold')
plt.ylabel('Dimension 2', fontweight='bold')
plt.legend()


plt.tight_layout()
plt.savefig('doc/fig.tsne.jpg', dpi=600, bbox_inches='tight',transparent=True)  # 高DPI版本
plt.savefig('doc/fig.tsne.svg', bbox_inches='tight', transparent=True)  # 矢量图版本