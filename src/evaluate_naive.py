import os
import pickle
import math
import click as ck
import numpy as np
import pandas as pd
from collections import deque, Counter
from matplotlib import pyplot as plt
from uutils import FUNC_DICT, Ontology, NAMESPACES


@ck.command()
@ck.option(
    '--root-data-dir', '-rtrdf', default='data/',
    help='Data file with training features')
@ck.option(
    '--datasets-name', '-dname', default='homo',
    help='Test data file')
@ck.option(
    '--ont', '-o', default='bp',
    help='GO subontology (bp, mf, cc)')
def main(root_data_dir, datasets_name, ont):

    go_rels = Ontology(root_data_dir + datasets_name + '/go.obo', with_rels=True)
    train_data_file = root_data_dir + datasets_name + f'/train_df.pkl'
    test_data_file = root_data_dir + datasets_name + f'/{ont}/test_df.pkl'
    terms_file = root_data_dir + datasets_name + f'/{ont}/{ont}.txt'
    F_txt = open(f'models/{datasets_name}/{ont.upper()}/{ont.upper()}_Naive_result', 'a+')

    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))

    go_rels.calculate_ic(annotations + test_annotations)

    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    annotations = list(map(lambda x: set(filter(lambda y: y in go_set, x)), annotations))

    cnt = Counter()
    max_n = 0
    for x in annotations:
        cnt.update(x)

    max_n = cnt.most_common(1)[0][1]

    scores = {}
    for go_id, n in cnt.items():
        score = n / max_n
        scores[go_id] = score

    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

    fmax = 0.0
    tmax = 0.0
    smin = 1000.0
    precisions = []
    recalls = []
    for t in range(101):
        threshold = t / 100.0
        preds = []
        annots = set()
        for go_id, score in scores.items():
            if score >= threshold:
                annots.add(go_id)
        for i, row in enumerate(test_df.itertuples()):
            preds.append(annots.copy())

        fscore, prec, rec, s = evaluate_annotations(go_rels, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')
    print(f'datasets:{datasets_name}\t Ont:{ont}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}', file=F_txt)
    F_txt.close()

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = real_annots[i].intersection(pred_annots[i])
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
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
    return f, p, r, s

if __name__ == '__main__':
    main()