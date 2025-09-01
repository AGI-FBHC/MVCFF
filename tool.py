import os
import pickle
import numpy as np

# ========== One-hot 编码函数 ==========
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def to_onehot(seq):
    mat = np.zeros((len(seq), 20), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            mat[i, aa_to_idx[aa]] = 1.0
    return mat


# ========== 读取FASTA ==========
def read_fasta(path):
    seqs = {}
    with open(path, "r") as f:
        seq_id, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    seqs[seq_id] = "".join(seq)
                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if seq_id is not None:
            seqs[seq_id] = "".join(seq)
    return seqs


# ========== 占位 PSSM 生成 ==========
def generate_dummy_pssm(seq_len, score=1.0):
    return np.ones((seq_len, 20), dtype=np.float32) * score


# ========== 占位 BERT 生成 ==========
def generate_dummy_bert(seq_len, hidden_dim=1024):
    return np.random.rand(seq_len, hidden_dim).astype(np.float32)


# ========== 构建特征 .pkl ==========
def build_feat_file(seq_id, seq, feats_dir, hidden_dim=128):
    os.makedirs(feats_dir, exist_ok=True)

    L = len(seq)
    # 占位 BERT embedding
    bert_emb = generate_dummy_bert(L, hidden_dim)
    # 占位 PSSM
    pssm = generate_dummy_pssm(L)

    feat_dict = {
        "bert": bert_emb,
        "pssm": pssm
    }

    save_path = os.path.join(feats_dir, f"{seq_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(feat_dict, f)
    print("✅ Saved", save_path)


# ========== 主函数 ==========
if __name__ == "__main__":
    base_dir = "./src/data/cafa3"
    feats_dir = os.path.join(base_dir, "feats_dir")

    for split in ["train", "test"]:
        fasta_path = os.path.join(base_dir, f"{split}.fa")
        seqs = read_fasta(fasta_path)

        for pid, seq in seqs.items():
            print(f"Processing {pid} ...")
            build_feat_file(pid, seq, feats_dir)
