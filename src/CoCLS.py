import argparse
from utils import *
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from networks import DeepCALayer, Co_attention, init_weights
from datasets1 import CNN1DDataset, cnn1d_collate
from model import load_checkpoint
from torch.utils.data import DataLoader as pyDataLoader
from data_utils import get_data, pass_config
import torch.nn.functional as F
import torch.optim as optim
from evaluation import compute_roc, compute_fmax, evaluate_annotations
from config import opt


def evaluate(device, model, criterion, contrast_model,eval_loader):
    model.eval()
    avg_loss = 0.0
    print("---Validating--- ")
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():  # set all 'requires_grad' to False
        pbar = tqdm(eval_loader)
        for data in pbar:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y
            z, z1, z2 = model(data)
            k = torch.tensor(int(z.shape[0] * 0.5), dtype=torch.float)
            p = (1 / torch.sqrt(k)) * torch.randn(int(k), z.shape[0]).to(device)

            z1 = p @ z1
            z2 = p @ z2
            h1, h2 = [model.project(x) for x in [z1, z2]]
            loss = contrast_model(h1, h2)

            z = torch.sigmoid(z)
            bce_loss = criterion(z, labels.float())
            loss = 0.0 * loss + bce_loss
            avg_loss += loss.item() / len(eval_loader)
            y_true.append(labels.cpu().numpy().squeeze())
            y_pred_sigm.append(z.cpu().numpy().squeeze())
            pbar.set_postfix({'loss': round(loss.item(), 4)})
            pbar.update()

        # Calculate evaluation metrics
        y_true = np.vstack(y_true)
        y_pred_sigm = np.vstack(y_pred_sigm)

        avg_auc = compute_roc(y_true, y_pred_sigm)
        # Maximum F-score
        avg_fmax = compute_fmax(y_true, y_pred_sigm, nrThresholds=10)

    return avg_loss,avg_auc,avg_fmax


class MVFF(nn.Module):
    def __init__(self, input_dims, num_classes, hidden_dim, type=None, n_view=3):
        super(MVFF, self).__init__()
        self.type = type
        self.n_view = n_view
        self.sub_nets = nn.ModuleList([DeepCALayer(input_dim=input_dims[i], hidden_dim=hidden_dim)
                                       for i in range(n_view)])

        self.co_attn1 = Co_attention(hidden_dim, r=4)
        self.co_attn2 = Co_attention(hidden_dim, r=4)

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc_out = nn.Sequential(nn.Linear(hidden_dim, num_classes),nn.ReLU())
        init_weights(self)

    def forward(self, data):
        # bert onehot pssm
        x = [data.x, data.x1, data.x2, data.x3]

        all_out = []
        for id, net in enumerate(self.sub_nets):
            conv_attn = net(x[id])
            all_out.append(conv_attn)

        [bert, onehot, pssm] = all_out
        bo_feat = self.co_attn1(bert, onehot)
        bp_feat = self.co_attn2(bert, pssm)
        all_out = [bert, bo_feat, bp_feat]
        final_out = []
        for x in all_out:
            output = torch.flatten(self.maxpool(x), 1)
            final_out.append(self.fc_out(output))

        return final_out


class Co_CLS(nn.Module):
    def __init__(self, encoder, num_classes, proj_dim):
        super(Co_CLS, self).__init__()
        self.encoder = encoder

        self.fc1 = nn.Linear(num_classes, proj_dim)
        self.fc2 = nn.Linear(proj_dim, num_classes)



    def forward(self, data):
        [z, z1, z2] = self.encoder(data)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5


class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss


def pser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--phase', dest='phase', default='train')  # 'train' / 'test'
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=12)  # cafa : 15 / homo : 30

    parser.add_argument('--namespace', dest='namespace', default="mf")
    parser.add_argument('--datasets', dest='datasets', default="cafa3")  # cafa / homo
    parser.add_argument('--net_type', dest='net_type', default="MVFFNet")  # DMPF / MLP /DeepGOCNN / DeepGOA /
    parser.add_argument('--feats_type', dest='feats_type', default="B_O_P")  # bert onehot pssm word2vec / B_O_P_W
    parser.add_argument('--namespace_dir', dest='namespace_dir', default="data/")
    parser.add_argument('--feats_dir', dest='feats_dir', default='E:/Datasets/cafa_feats')  # cafa_feats / homo_feats
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--out_file', dest='out_file')
    return parser.parse_args()


def main(**kwargs):
    opt.parse(kwargs)
    args = pser_args()
    EPOCHS = 12

    go = Ontology(f'data/{args.datasets}/go.obo', with_rels=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[*] Selected device:", device)
    print("[*] Using [%s] features." % args.feats_type)
    # dataPre
    input_dim, input_dims = pass_config(device, args, opt)
    terms_dict, terms, train_names, valid_names, test_names, train_df, test_df, num_classes = get_data(
        'data/' + args.datasets, args.namespace)

    train_names = train_names[:1000]

    encoder = MVFF(input_dims=input_dims, num_classes=num_classes, hidden_dim=512, type=None).to(device)
    args.model_dir = "models/" + args.datasets + "/" + args.namespace.upper() + "/" + args.net_type + '/' + args.feats_type + '/checkpoint' + f'/model_test.pth.tar'
    load_checkpoint(encoder, filename=args.model_dir)

    train_set = CNN1DDataset(names=train_names, feats_dir=args.feats_dir, terms_dict=terms_dict,
                             feats_type=args.feats_type)
    train_loader = pyDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                collate_fn=cnn1d_collate)

    valid_set = CNN1DDataset(names=valid_names, feats_dir=args.feats_dir, terms_dict=terms_dict,
                             feats_type=args.feats_type)
    valid_loader = pyDataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=cnn1d_collate)

    test_set = CNN1DDataset(names=test_names, feats_dir=args.feats_dir, terms_dict=terms_dict,
                            feats_type=args.feats_type)
    test_loader = pyDataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                               collate_fn=cnn1d_collate)

    model = Co_CLS(encoder, num_classes, 256).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCE(
        tau=0.4), mode='L2L', intraview_negs=True).to(device)
    filename = f'models/{args.datasets}/{args.namespace.upper()}/Contrastive/Con_best.pth.tar'
    criterion = nn.BCELoss().to(device)
    if args.phase == 'train':
        model.train()
        best_epoch = 0
        best_loss = 10
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        for epoch in range(EPOCHS):
            # train
            print(f"Training {epoch}")
            pbar = tqdm(train_loader)
            for data in pbar:
                data = data.to(device)
                label = data.y
                z, z1, z2 = model(data)

                k = torch.tensor(int(z.shape[0] * 0.5), dtype=torch.float)
                p = (1 / torch.sqrt(k)) * torch.randn(int(k), z.shape[0]).to(device)

                z1 = p @ z1
                z2 = p @ z2
                h1, h2 = [model.project(x) for x in [z1, z2]]

                loss = contrast_model(h1, h2)
                z = torch.sigmoid(z)
                bce_loss = criterion(z,label.float())
                loss = 0.0 * loss + bce_loss
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

            valid_loss,valid_auc,valid_fmax = evaluate(device, model, criterion,contrast_model, valid_loader)
            print("--- valid Loss:                  %.4f" % valid_loss)
            print("--- valid roc auc score:         %.4f" % valid_auc)
            print("--- valid max F-score:           %.4f" % valid_fmax)
            if valid_loss < best_loss:
                best_loss = valid_loss
                print("--- Best_epoch:    {:.2f}\t          Best Loss:{:.4f}".format(best_epoch, best_loss))
                # Save last model
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                torch.save(state, filename)
    else:
        try:
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
        except:
            print("[!] No checkpoint found, start epoch 0")
        model.eval()
        avg_loss = 0.0
        y_true = []
        y_pred_sigm = []

        with torch.no_grad():  # set all 'requires_grad' to False
            pbar = tqdm(test_loader)
            for data in pbar:
                # Get current batch and transfer to device
                data = data.to(device)
                labels = data.y
                outputs, _, _ = model(data)
                outputs = torch.sigmoid(outputs)
                current_loss = criterion(outputs, labels.float())
                avg_loss += current_loss.item() / len(test_loader)

                y_true.append(labels.cpu().numpy().squeeze())
                y_pred_sigm.append(outputs.cpu().numpy().squeeze())
                pbar.set_postfix({'loss': round(current_loss.item(), 4)})
                pbar.update()

            # Calculate evaluation metrics
            # y_true = np.vstack(y_true)
            y_pred_sigm = np.vstack(y_pred_sigm)

            # roc_auc = compute_roc(y_true, y_pred_sigm)
            # # # Maximum F-score
            # avg_fmax = compute_fmax(y_true, y_pred_sigm, nrThresholds=10)

        annotations = train_df['prop_annotations'].values
        annotations = list(map(lambda x: set(x), annotations))
        test_annotations = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for go_id in row.prop_annotations:
                if go.has_term(go_id):
                    annots |= go.get_anchestors(go_id)
            test_annotations.append(annots)

        go.calculate_ic(annotations + test_annotations)
        go_set = go.get_namespace_terms(NAMESPACES[args.namespace])
        go_set.remove(FUNC_DICT[args.namespace])

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
            for i, row in enumerate(test_df.itertuples()):
                annots = set()
                for j, score in enumerate(y_pred_sigm[i]):
                    if score >= threshold:
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
                test_id = list(preds[54])
                test_id.sort()
                # for name in test_id:
                #     print(name.strip())
            if smin > s:
                smin = s

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)
        print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')


if __name__ == '__main__':
    main()
