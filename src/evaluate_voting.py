import numpy as np
import pandas as pd
import click as ck
import logging
import pickle
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import math
import time
import random
import torch
from evaluation import compute_fmax
from utils import FUNC_DICT, Ontology, NAMESPACES
from collections import Counter
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

@ck.command()
@ck.option(
    '--root-data-dir', '-rtrdf', default='data/',
    help='Data file with training features')
@ck.option(
    '--datasets-name', '-dname', default='cafa3',
    help='Test data file')
@ck.option(
    '--ont', '-o', default='bp',
    help='GO subontology (bp, mf, cc)')
def main(root_data_dir, datasets_name, ont):
    net_type = 'DeepCNN'
    train_data_file = root_data_dir + datasets_name + f'/train_df.pkl'
    test_data_file = root_data_dir + datasets_name + f'/{ont}/test_df.pkl'
    terms_file = root_data_dir + datasets_name + f'/{ont}/{ont}.txt'
    F_txt = open(f'models/{datasets_name}/{ont}/{ont.upper()}_{net_type}_Soft_Voting.txt','a+')
    feats_list = ['word2vec','bert','pssm','onehot']
    num_feats = len(feats_list)
    go = Ontology(root_data_dir + datasets_name+'/go.obo', with_rels=True)

    terms = np.loadtxt(terms_file, dtype=str)
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)

    print("Length of test set: " + str(len(test_df)))
    # Annotations
    test_annotations = []
    for i, row in enumerate(test_df.itertuples()):
        annots = set()
        for go_id in row.prop_annotations:
            if go.has_term(go_id):
                annots |= go.get_anchestors(go_id)
        test_annotations.append(annots)
    test_labels = np.zeros((len(test_df), len(terms)), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                test_labels[i, terms_dict[go_id]] = 1
    go.calculate_ic(annotations + test_annotations)

    # DeepGO
    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    pred_df = pd.DataFrame(columns=['proteins', 'y_true'])
    pred_df['proteins'] = test_df['proteins']
    pred_df['y_true'] = labels
    alphas = [0.05]
    pred_list = []
    for pred_name in feats_list:
        predictions = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/{pred_name}/prediction.pkl', 'rb'))['y_pred']
        pred_list.append(predictions)

    def get_sum_threshold(a_pred,b_pred,labels):
        fscore_max = 0
        x = 0
        count = 0
        for i in range(101):
            alpha = i / 100.0

            predictions = alpha * a_pred + (1 - alpha) * b_pred
            temp_fscore_max = compute_fmax(labels, predictions, 10)
            count +=1
            if temp_fscore_max > fscore_max:
                count = 0
                x = alpha
                fscore_max = temp_fscore_max

            if count ==10:
                break
        return x

    left_pred = pred_list[0]
    # left_pred = o_predictions * 0.42 + 0.58 * w_predictions
    alphas = []
    for i in range(num_feats-1):
        t = get_sum_threshold(left_pred,pred_list[i+1],test_labels)
        left_pred = t * left_pred + (1-t) * pred_list[i+1]
        print(f'id:{i}\t alpha:{t}')
        alphas.append(t)


    fusion_prediction = left_pred
    # fusion_prediction = 0.05 * w_predictions + 0.95 * b_predictions
    def get_metrics(predictions):

        fa = 0.0
        tmax = 0.0
        smin = 1000.0
        precisions = []
        recalls = []
        for t in range(101):
            threshold = t / 100.0
            preds = []
            for i, row in enumerate(test_df.itertuples()):
                annots = set()
                for j, score in enumerate(predictions[i]):
                    if score >=threshold:
                        annots.add(terms[j])
                new_annots = set()
                for go_id in annots:
                    new_annots |= go.get_anchestors(go_id)
                preds.append(new_annots)

            # Filter classes
            preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))

            fscore, prec, rec, s = evaluate_annotations(go, labels, preds)
            precisions.append(prec)
            recalls.append(rec)
            print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
            if fa < fscore:
                fa = fscore
                tmax = threshold
                pred_df['Voting'] = preds

            if smin > s:
                smin = s
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)
        pred_df.to_pickle(f'case_study/Voting.pkl')
        return fa,smin,aupr,tmax


    print(f"{feats_list} \t Voting\t{ont.upper()}")
    print(f"{feats_list} \t Voting\t{ont.upper()}",file=F_txt)

    start_time = time.time()

    fscore, smin, aupr, threshold = get_metrics(fusion_prediction)
    end_time = time.time()

    print(
        f'alphas:{alphas}\tFmax: {fscore:0.3f}, Smin: {smin:0.3f}, Aupr: {aupr:0.3f}, threshold: {threshold} time:{(end_time - start_time) / 60.0}')
    print(
        f'alphas:{alphas}\tFmax: {fscore:0.3f}, Smin: {smin:0.3f}, Aupr: {aupr:0.3f}, threshold: {threshold} time:{(end_time - start_time) / 60.0}',
        file=F_txt)


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
