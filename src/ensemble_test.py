import numpy as np
import pandas as pd
import click as ck
import logging
import pickle
import torch
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from collections import Counter
from matplotlib import pyplot as plt

preds_1 = pickle.load(open('models/homo/MF/DeepGOCNN/bert/prediction.pkl', 'rb'))
predictions_1 = preds_1['y_pred']   # (3082, 398)
preds_2 = pickle.load(open('models/homo/MF/DeepGOCNN/pssm/prediction.pkl', 'rb'))
predictions_2 = preds_2['y_pred']
preds_3 = pickle.load(open('models/homo/MF/DeepGOCNN/word2vec/prediction.pkl', 'rb'))
predictions_3 = preds_3['y_pred']
preds_4 = pickle.load(open('models/homo/MF/DeepGOCNN/onehot/prediction.pkl', 'rb'))
predictions_4 = preds_4['y_pred']

terms = np.loadtxt('data/homo/mf/mf.txt',dtype=str)
nb_classes = len(terms)
terms_dict = {v: i for i, v in enumerate(terms)}
test_df = pickle.load(open('data/homo/mf/test_df.pkl','rb'))
test_labels = np.zeros((len(test_df), nb_classes), dtype=np.int32)
for i, row in enumerate(test_df.itertuples()):
    for go_id in row.prop_annotations:
        if go_id in terms_dict:
            test_labels[i, terms_dict[go_id]] = 1

print(test_labels.shape)   # (3082,398)

criterion = torch.nn.BCELoss()


def cross_entropy_error(y,t):
    #y是一维的情况
    if y.ndim==1:
        #转为二维：shape是(1,t.size)的类型(这里是1,10)，
        #而不是一维shape(t.size,)的情况(这里是10，)了
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size #就是又多少行（batch_size），就除以多少，这就是小批量来估计整体

y1 = cross_entropy_error(predictions_1,test_labels)
y2 = cross_entropy_error(predictions_2,test_labels)
y3 = cross_entropy_error(predictions_3,test_labels)
y4 = cross_entropy_error(predictions_4,test_labels)


# 误差
# y1 = np.linalg.norm(test_labels-predictions_1)
# y2 = np.linalg.norm(test_labels-predictions_2)
# y3 = np.linalg.norm(test_labels-predictions_3)
# y4 = np.linalg.norm(test_labels-predictions_4)

print(y1)
print(y2)
print(y3)
print(y4)
# y5 = np.linalg.norm(test_labels-(0.5 * predictions_1 + 0.5 * predictions_4))



min_error = y1
tmax_1 = 0
tmax_2 = 0
tmax_3 = 0
tmax_4 = 0

initial_value = int((1 / 4.0) * 100)
alpha_list = []
for t in range(100,initial_value,-1):
    alpha_1 = t / 100.0

    for n in range(t):
        alpha_2 = n / 100.0

        for m in range(n+1):
            alpha_3 = m / 100.0
            for k in range(m+1):
                alpha_4 = k /100.0
                if alpha_1 + alpha_2 + alpha_3 + alpha_4 == 1:
                    predictions = alpha_1 * predictions_1 + alpha_2 * predictions_2 + alpha_3 * predictions_3 + alpha_4 * predictions_4
                    error = cross_entropy_error(predictions,test_labels)
                    if error < min_error:
                        print(f'a:{alpha_1},b:{alpha_2},c:{alpha_3},d:{alpha_4}->')
                        min_error = error
                        tmax_1 = alpha_1
                        tmax_2 = alpha_2
                        tmax_3 = alpha_3
                        tmax_4 = alpha_4
                        alpha_list.append([tmax_1,tmax_2,tmax_3,tmax_4])
                        print(f"new_lowerest_error: {error}")
#
print(min_error)
print(tmax_1)
print(tmax_2)
print(tmax_3)
print(tmax_4)

