import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')

# 读取图片
img = np.array(Image.open("doc/img/O75821.png"))

# 假设有人的注视点数据 [(x1, y1), (x2, y2), ...]

# gaze_points = np.array([[45, 100],[50, 90],
#                         [80, 60],[80, 45], [90, 50], [90, 30], [100, 40],[100, 60], [110, 30],[110, 40],[110, 50], [120, 50],[120, 30]])

gaze_points = np.array([
                        # [150, 150],[150, 180],[140,170],[120,150],[120,180],[170,160],
                        # [80,200],[100,200],
                        # [100,590],[100,600],[100,620],[100,630],[90,610],[70,630],[60,620],[50,610],
                        # [250,500],[250,520], [250,530],[250,550],[250,570],[250,580],[260,560],
                        # [200,80],[200,50],[220,70],[220,40],
                        # [300,350],[320,340],[330,330],[350,350],
                        # [400,200],[400,230],[400,250],[380,190],[380,200],[380,220],[380,250],[380,270],
                        [650,400],[660,410],
                        # [750,100],[770,100],[800,100],[770,120],[750,120],[800,120],
                        # [800,300],[800,270],[800,250],[820,280],[820,220],[780,280],
                        [900,500],[920,520],[940,540],[950,590],[950,450],[920,450],[970,480],[950,480],[950,470],[920,470],[970,530],[990,530],[990,510],
                        [1030,570],[1050,570]
                        ])

# 创建热图矩阵
heatmap = np.zeros((img.shape[0], img.shape[1])) 
for x, y in gaze_points:
    heatmap[y, x] += 1

# 平滑热图
heatmap = gaussian_filter(heatmap, sigma=15)


# 设置坐标轴范围为图片尺寸（像素）
# plt.xlim(0, img.shape[1])
# plt.ylim(img.shape[0], 0)  # y轴反转，使原点在左上角
# plt.xticks(np.arange(0, img.shape[1]+1, 50))  # 设置x轴刻度间隔
# plt.yticks(np.arange(0, img.shape[0]+1, 50))  # 设置y轴刻度间隔
# plt.xlabel("X (pixels)")
# plt.ylabel("Y (pixels)")
# plt.grid(color='r', linestyle='--', linewidth=0.5)

# 可视化
plt.imshow(img)
plt.imshow(heatmap, cmap='jet', alpha=0.7)  # alpha 控制透明度
plt.axis('off')




# 调整布局
plt.tight_layout()

# 保存图像
# plt.savefig('fig.performance_comparison.png', dpi=600, bbox_inches='tight')
plt.savefig('doc/fig.heatmap.jpg', dpi=600, bbox_inches='tight', transparent=True)
# plt.savefig('fig.performance_comparison.pdf', bbox_inches='tight')
plt.savefig('doc/fig.heatmap.svg', bbox_inches='tight', transparent=True)