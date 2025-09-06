import os
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# 1. 导入模型
# -----------------------------
from networks import MVFF  # 替换成你的模型路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 四种特征维度
input_dims = [21, 14, 21, 1280]  # Onehot, OPF, PSSM, ESM
hidden_dim = 128
num_classes = 1  # 根据你的任务类别数

model = MVFF(
    device=device,
    input_dims=input_dims,
    num_classes=num_classes,
    hidden_dim=hidden_dim
)
# 加载训练好的模型（如果有）
# model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. 加载 pickle 特征文件
# -----------------------------
protein_id = "A0A060X6Z0"
base_dir = "D:/project/PythonProject/single_protein_feats"  # 修改成你存 pickle 的目录

def load_pickle_feat(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# 读取特征
x_onehot = torch.tensor(load_pickle_feat(os.path.join(base_dir, f"{protein_id}_onehot.npz")), dtype=torch.float32).unsqueeze(0).to(device)
x_opf    = torch.tensor(load_pickle_feat(os.path.join(base_dir, f"{protein_id}_opf.npz")), dtype=torch.float32).unsqueeze(0).to(device)
x_pssm   = torch.tensor(load_pickle_feat(os.path.join(base_dir, f"{protein_id}_pssm.npz")), dtype=torch.float32).unsqueeze(0).to(device)
x_esm    = torch.tensor(load_pickle_feat(os.path.join(base_dir, f"{protein_id}_esm.npz")), dtype=torch.float32).unsqueeze(0).to(device)

# 转换为 [batch, channels, length]
x_onehot = x_onehot.permute(0, 2, 1)  # [1, L, 21] -> [1, 21, L]
x_opf    = x_opf.permute(0, 2, 1)     # [1, L, 14] -> [1, 14, L]
x_pssm   = x_pssm.permute(0, 2, 1)    # [1, L, 20] -> [1, 20, L]
x_esm    = x_esm.permute(0, 2, 1)     # [1, L, 1280] -> [1, 1280, L]

def fix_channels(x, expected_c):
    if x.shape[1] < expected_c:
        pad = torch.zeros((x.shape[0], expected_c - x.shape[1], x.shape[2]), device=x.device)
        x = torch.cat([x, pad], dim=1)
    return x

x_onehot = fix_channels(x_onehot, 21)
x_opf    = fix_channels(x_opf, 14)
x_pssm   = fix_channels(x_pssm, 21)
x_esm    = fix_channels(x_esm, 1280)
data = {
    "x": x_onehot,
    "x1": x_opf,
    "x2": x_pssm,
    "x3": x_esm
}

print("Feature shapes:")
print("Onehot:", x_onehot.shape)
print("OPF:", x_opf.shape)
print("PSSM:", x_pssm.shape)
print("ESM:", x_esm.shape)

# -----------------------------
# 3. 前向计算 & 提取 Attention
# -----------------------------
with torch.no_grad():
    enc_outs = []
    for i, enc_layer in enumerate(model.sub_nets):
        conv_attn = enc_layer(data[f"x{i}" if i > 0 else "x"])
        enc_outs.append(conv_attn)

    # 公共特征
    common_feat_ini = model.common_project_1(torch.cat(enc_outs, dim=1))
    common_feat_tep = model.common_project_2(common_feat_ini)
    common_feat = common_feat_tep * model.cross_linear(common_feat_tep) + common_feat_ini

    # 跨视角注意力
    attn_list = []
    for co_layer, sub_x in zip(model.co_layers, enc_outs):
        enc_attn = co_layer(common_feat, sub_x)
        attn_list.append(enc_attn)

    # 融合后的注意力特征
    attn_feat = torch.cat(attn_list, dim=1).squeeze(0).cpu().numpy()  # shape: [hidden_dim*n_view, L]

# -----------------------------
# 4. 可视化 Attention
# -----------------------------
plt.figure(figsize=(12, 4))
sns.heatmap(attn_feat, cmap="viridis", cbar=True)
plt.xlabel("Residue position")
plt.ylabel("Attention channels")
plt.title(f"Four-view Attention Map")
plt.tight_layout()

save_path = os.path.join(base_dir, f"{protein_id}_attention_map.svg")
plt.savefig(save_path, dpi=600)  # dpi 可调，300 为高质量打印
plt.show()

print(f"Attention map saved to: {save_path}")