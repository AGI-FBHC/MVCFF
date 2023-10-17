import numpy as np
import pandas as pd
import click as ck
import logging
import pickle

from uutils import FUNC_DICT, Ontology, NAMESPACES,evaluate_annotations

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--root-data-dir', '-rtrdf', default='data/',
    help='Data file with training features')
@ck.option(
    '--datasets-name', '-dname', default='cafa3',
    help='Test data file')
@ck.option(
    '--ont', '-o', default='cc',
    help='GO subontology (bp, mf, cc)')
def main(root_data_dir, datasets_name, ont):
    net_type = 'DeepGOCNN'
    feats_list = ['onehot','pssm','bert']

    train_data_file = root_data_dir + datasets_name + f'/{ont}/train_df.pkl'
    test_data_file = root_data_dir + datasets_name + f'/{ont}/test_df.pkl'
    terms_file = root_data_dir + datasets_name + f'/{ont}/{ont}.txt'
    F_txt = open(f'models/{datasets_name}/{ont}/{ont}_{net_type}_majority_voting.txt','a+')

    go = Ontology(root_data_dir + datasets_name+'/go.obo', with_rels=True)

    majority_voting = pd.DataFrame(columns=feats_list)
    onehot_preds = pd.DataFrame(pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/onehot/prediction.pkl', 'rb'))['y_pred'])
    pssm_preds = pd.DataFrame(pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/pssm/prediction.pkl', 'rb'))['y_pred'])
    bert_preds = pd.DataFrame(pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/bert/prediction.pkl', 'rb'))['y_pred'])


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

    go.calculate_ic(annotations + test_annotations)

    # DeepGO
    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

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
        print(threshold)
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            data_row = pd.DataFrame(pd.concat([onehot_preds.iloc[[i]], pssm_preds.iloc[[i]], bert_preds.iloc[[i]]], axis=0))
            data_row[data_row >= threshold] = 1
            data_row[data_row < threshold] = 0
            vote = data_row.mode()
            for j in range(vote.shape[1]):
                if vote[j].item() == 1:
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
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s


    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'datasets:{datasets_name}\t Ont:{ont}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}',file=F_txt)
    F_txt.close()









if __name__ == '__main__':
    main()

# feats_list = ['onehot','pssm']
# onehot = pickle.load(open(f'models/cafa3/MF/DeepCNN/onehot/prediction.pkl', 'rb'))
# onehot = pd.DataFrame(onehot['y_pred'])
# pssm = pickle.load(open(f'models/cafa3/MF/DeepCNN/pssm/prediction.pkl', 'rb'))
# pssm = pd.DataFrame(pssm['y_pred'])
# majority_voting = pd.DataFrame([])
# for x in range(len(onehot)):
#
#     data_row = pd.DataFrame(pd.concat([onehot.iloc[[x]],pssm.iloc[[x]]], axis = 0 ))
#     vote = data_row.mode()
#     majority_voting = majority_voting.append(vote)
#
# print(majority_voting)

