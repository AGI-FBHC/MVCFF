import numpy as np
import pickle
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from uutils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


datasets = 'cafa3'
@ck.command()
@ck.option(
    '--train-data-file', '-trdf', default=f'data/{datasets}/train_df.pkl',
    help='Data file with training features')
@ck.option(
    '--diamond-scores-file', '-dsf', default=f'data/{datasets}/test_diamond.res',
    help='Diamond output')
@ck.option(
    '--ont', '-o', default='bp',
    help='GO subontology (bp, mf, cc)')
def main(train_data_file, diamond_scores_file, ont):

    net_type = 'DeepCNN'
    datasets_name = 'cafa3'
    feat_type = 'bert'
    go_rels = Ontology(f'data/{datasets}/go.obo', with_rels=True)

    terms_file = f'data/{datasets}/{ont}/{ont}.txt'
    terms = np.loadtxt(terms_file, dtype=str)
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()
    valid_data_file = f'data/{datasets}/{ont}/test_df.pkl'

    # deep_predictions = pickle.load(open(f'models/{datasets}/{ont.upper()}/{net_type}/{feat_type}/prediction.pkl','rb'))['y_pred']

    o_predictions = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/onehot/prediction.pkl', 'rb'))['y_pred']
    w_predictions = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/word2vec/prediction.pkl', 'rb'))[
        'y_pred']
    p_predictions = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/pssm/prediction.pkl', 'rb'))['y_pred']
    b_predictions = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/bert/prediction.pkl', 'rb'))['y_pred']
    alphas = [0.42, 0.21, 0.5]
    deep_predictions = alphas[2] * (alphas[1] * (alphas[0] * o_predictions + (1 - alphas[0]) * w_predictions) + (
                1 - alphas[1]) * p_predictions) + (1 - alphas[2]) * b_predictions

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    valid_annotations = valid_df['prop_annotations'].values
    valid_annotations = list(map(lambda x: set(x), valid_annotations))
    go_rels.calculate_ic(annotations + valid_annotations)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    
    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])

    blast_preds = []
    for i, row in enumerate(valid_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= annotations[prot_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[prot_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            ind = np.argsort(-sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)
        
    # DeepGOPlus
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = valid_df['prop_annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    best_fmax = 0.0
    best_alpha = 0.0
    print(f'datasets:{datasets.upper()}\tont:{ont.upper()}')
    for alpha in range(20, 70):
        alpha /= 100.0
        deep_preds = []
        for i, row in enumerate(valid_df.itertuples()):
            annots_dict = blast_preds[i].copy()
            if bool(annots_dict):
                for go_id in annots_dict:
                    annots_dict[go_id] *= alpha
                for j, score in enumerate(deep_predictions[i]):
                    go_id = terms[j]
                    score *= 1 - alpha
                    if go_id in annots_dict:
                        annots_dict[go_id] += score
                    else:
                        annots_dict[go_id] = score

            else:
                for j, score in enumerate(deep_predictions[i]):
                    go_id = terms[j]
                    annots_dict[go_id] = score

            deep_preds.append(annots_dict)

        fmax = 0.0
        tmax = 0.0
        precisions = []
        recalls = []
        smin = 1000000.0
        rus = []
        mis = []
        for t in range(10, 30):
            threshold = t / 100.0
            preds = []
            for i, row in enumerate(valid_df.itertuples()):
                annots = set()
                for go_id, score in deep_preds[i].items():
                    if score >= threshold:
                        annots.add(go_id)

                new_annots = set()
                for go_id in annots:
                    new_annots |= go_rels.get_anchestors(go_id)
                preds.append(new_annots)

            # Filter classes
            preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))

            fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, labels, preds)
            avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
            avg_ic = sum(map(lambda x: sum(map(lambda go_id: go_rels.get_ic(go_id), x)), fps)) / len(fps)
            # print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
            if fmax < fscore:
                fmax = fscore
                tmax = threshold
            if smin > s:
                smin = s
        if best_fmax < fmax:
            best_fmax = fmax
            best_alpha = alpha
        print(f'Alpha: {alpha} Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    print(f'{best_alpha} {best_fmax}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns


if __name__ == '__main__':
    main()
