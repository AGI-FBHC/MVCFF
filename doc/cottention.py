import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from networks import MVFF

# =========================== # 配置 # ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dims = [21, 14, 21, 1280]
hidden_dim = 128
num_classes = 1
model = MVFF(device=device, input_dims=input_dims, num_classes=num_classes, hidden_dim=hidden_dim)
model.to(device)
model.eval()

# 数据路径
base_dir = "E:/Datasets/cafa3"
feat_dirs = { "onehot": os.path.join(base_dir, "onehot"),
              "opf": os.path.join(base_dir, "opf"),
              "pssm": os.path.join(base_dir, "pssm"),
              "esm": os.path.join(base_dir, "esm"), }
save_dir = os.path.join(base_dir, "attn_results")
os.makedirs(save_dir, exist_ok=True)

# =========================== # 工具函数 # ===========================
def load_npz_feat(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        arr = data[list(data.keys())[0]]
    else:
        arr = data
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = arr.item()
        except Exception:
            pass
        return arr

def fix_channels(x, expected_c):
    if x.shape[1] < expected_c:
        pad = torch.zeros((x.shape[0], expected_c - x.shape[1], x.shape[2]), device=x.device)
        x = torch.cat([x, pad], dim=1)
    return x

def process_single_protein(protein_id):
    x_onehot = torch.tensor(load_npz_feat(os.path.join(feat_dirs["onehot"], f"{protein_id}.npz")),
                            dtype=torch.float32).unsqueeze(0).to(device)
    x_opf = torch.tensor(load_npz_feat(os.path.join(feat_dirs["opf"], f"{protein_id}.npz")),
                         dtype=torch.float32).unsqueeze(0).to(device)
    x_pssm = torch.tensor(load_npz_feat(os.path.join(feat_dirs["pssm"], f"{protein_id}.npz")),
                          dtype=torch.float32).unsqueeze(0).to(device)
    x_esm = torch.tensor(load_npz_feat(os.path.join(feat_dirs["esm"], f"{protein_id}.npz")),
                         dtype=torch.float32).unsqueeze(0).to(device)

    x_onehot = fix_channels(x_onehot.permute(0, 2, 1), 21)
    x_opf = fix_channels(x_opf.permute(0, 2, 1), 14)
    x_pssm = fix_channels(x_pssm.permute(0, 2, 1), 21)
    x_esm = fix_channels(x_esm.permute(0, 2, 1), 1280)

    data = {"x": x_onehot, "x1": x_opf, "x2": x_pssm, "x3": x_esm}

    with torch.no_grad():
        enc_outs = []
        for i, enc_layer in enumerate(model.sub_nets):
            conv_attn = enc_layer(data[f"x{i}" if i > 0 else "x"])
            enc_outs.append(conv_attn)

        common_feat_ini = model.common_project_1(torch.cat(enc_outs, dim=1))
        common_feat_tep = model.common_project_2(common_feat_ini)
        common_feat = common_feat_tep * model.cross_linear(common_feat_tep) + common_feat_ini

        attn_list = []
        for co_layer, sub_x in zip(model.co_layers, enc_outs):
            enc_attn = co_layer(common_feat, sub_x) # [B, hidden_dim, L]
            attn_list.append(enc_attn.mean(dim=2)) # 对长度维取平均，得到 [B, hidden_dim]
            attn_feat = torch.cat(attn_list, dim=1).squeeze(0).cpu().numpy() # [n_views*hidden_dim]

    return attn_feat

# =========================== # 主逻辑：统计 & 作图 # ===========================
if __name__ == "__main__":
    data = pd.read_pickle("data/cafa3/test_data.pkl")

    if isinstance(data, pd.DataFrame) and "proteins" in data.columns and "sequences" in data.columns:
        protein_ids = data["proteins"].tolist() # 直接从 sequences 列计算长度
        seq_lens = data["sequences"].apply(len).tolist()
    elif isinstance(data, dict):
        protein_ids = list(data.keys())
        seq_lens = [len(v[0]) for v in data.values()] # v[0] 是序列
    else:
        raise ValueError("❌ 无法解析 test_data.pkl，请检查文件格式！")


    # 分箱
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 5000]
    bin_labels = ["50", "100", "150", "200", "250", "300", "400", "500", "750", "1000", "1250", "1500+"]

    attn_matrix = np.zeros((len(bin_labels), 4)) # [length_bin, n_views]
    counts = np.zeros(len(bin_labels))

    for idx, (protein_id, seq_len) in enumerate(zip(protein_ids, seq_lens)):
        try:
            print(f"[{idx+1}/{len(protein_ids)}] Processing {protein_id}...")
            attn_feat = process_single_protein(protein_id)
            # 分成四个视角
            n_views = 4
            view_attn = np.split(attn_feat, n_views)
            view_mean = [v.mean() for v in view_attn]

            bin_idx = np.digitize(seq_len, bins) - 1
            if 0 <= bin_idx < len(bin_labels):
                attn_matrix[bin_idx] += view_mean
                counts[bin_idx] += 1

        except Exception as e:
            print(f"❌ Failed: {protein_id} ({e})")

    attn_matrix = attn_matrix / np.maximum(counts[:, None], 1)

    # 作图
    plt.figure(figsize=(6, 5))
    df = pd.DataFrame(attn_matrix, index=bin_labels, columns=["OneHot","OPF","PSSM","ESM"])
    sns.heatmap(df, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Attention Weight'})
    plt.xlabel("Feature View")
    plt.ylabel("Sample Categories (Protein Length)")
    plt.title("Attention Weights Across Feature Views")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_heatmap.png"), dpi=600)
    plt.show()
    print("✅ 注意力热力图已保存到:", os.path.join(save_dir, "attention_heatmap.svg"))