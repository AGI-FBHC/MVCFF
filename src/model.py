import pickle
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluation import compute_fmax,compute_roc
from utils import now
import warnings
warnings.filterwarnings("ignore")

def train(opt,net, criterion, train_loader, valid_loader,test_loader,ckpt_dir, F_txt):
    device = opt.device
    lr_sched = opt.lr_sched
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    # Define scheduler for learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # Load checkpoint model and optimizer
    best_epoch = 0
    # best_epoch = load_checkpoint(net, optimizer, scheduler, filename=ckpt_dir + '/model_test.pth.tar')
    if best_epoch != 0:
        best_valid_loss, best_valid_auc,best_valid_fmax, _, _ = evaluate(opt,device,net,criterion,valid_loader)
        test_loss, test_auc,best_test_fmax, _, _ = evaluate(opt,device,net,criterion,test_loader)
        print(f"--- best_epoch : {best_epoch} \t best_Fmax : {best_valid_fmax} ---")
    else:
        best_valid_fmax = -1
        best_test_fmax = -1

    start_time = time.time()
    valid_loss_save = []
    for epoch in range(best_epoch,opt.num_epochs):
        net.train()
        # Print current learning rate
        print(f"[*] Epoch:{epoch+1} \t Namespace:{opt.namespace.upper()} \t net:{opt.net_type} \t feat:{opt.feats_type}" )
        for param_group in optimizer.param_groups:
            print('--- Current learning rate: ', param_group['lr'])
        pbar = tqdm(train_loader)
        for data in pbar:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y
            with torch.set_grad_enabled(True):
                # Set the parameter gradients to zero
                optimizer.zero_grad()
                # Forward pass
                outputs = net(data)
                current_loss = criterion(outputs, labels.float())
                # Backward pass and optimize
                current_loss.backward()
                optimizer.step()
            pbar.set_postfix({'loss': current_loss.item()})
            pbar.update()

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

        valid_loss, valid_auc,valid_fmax, _, _ = evaluate(opt,device,net,criterion,valid_loader)
        test_loss, test_auc,test_fmax, _, _ = evaluate(opt,device,net,criterion,test_loader)
        # print(f'{net.Wa.data}\t{net.Wc.data}')
        valid_loss_save.append(valid_loss)
        if valid_fmax > best_valid_fmax:
            best_epoch = epoch + 1
            best_valid_fmax = valid_fmax
            torch.save(state, ckpt_dir + '/model_best.pth.tar')

        if test_fmax > best_test_fmax:
            best_epoch = epoch + 1
            best_test_fmax = test_fmax
            torch.save(state, ckpt_dir + '/model_test.pth.tar')

        # print("Epoch:[{}]\t Valid loss:{:.4f}\t Valid AUPR:{:.4f}\t Valid ROC AUC:{:.4f}\t Valid Smin distance:{:.4f}\t Valid F-score:{:.4f}\t".format(epoch+1,avg_valid_loss,avg_valid_avgprec,roc_auc,avg_valid_sdmin,avg_valid_fmax),file=F_txt)
        print("Epoch:[{}]\t Valid loss:{:.4f}\t Valid ROC AUC:{:.4f}\t Valid F-score:{:.4f}\tTest F-score:{:.4f}".format(epoch+1,valid_loss,valid_auc,valid_fmax,test_fmax),file=F_txt)
        F_txt.flush()

        print("--- Best_epoch:    {:.2f}\t          Best Fmax:{:.4f}".format(best_epoch, best_valid_fmax))
        print("--- valid Loss:                  %.4f" % valid_loss)
        print("--- valid roc auc score:         %.4f" % valid_auc)
        print("--- valid max F-score:           %.4f" % valid_fmax)
        print("---  test max F-score:           %.4f" % test_fmax)

        # LR scheduler on plateau (based on validation loss)
        if lr_sched:
            scheduler.step(valid_loss)


    Loss = np.array(valid_loss_save)
    np.save(ckpt_dir+"Epoch_{}".format(opt.num_epochs),Loss)
    plot_loss(opt.num_epochs,save_dir=ckpt_dir)
    end_time = time.time()

    print("[*] Finish training.")
    print("*" * 20)
    print(f"{now()} finished; best_epoch: {best_epoch}; best_Fmax:{best_valid_fmax},time/epoch:{(end_time - start_time)/ opt.num_epochs}")
    print(f"{now()} finished; best_epoch: {best_epoch}; best_Fmax:{best_valid_fmax},time/epoch:{(end_time - start_time)/ opt.num_epochs}",file=F_txt)
    torch.cuda.empty_cache()
    return

def test(opt,device, net, criterion, model_file, test_loader,save_file=None,):
    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)

    # Evaluate model
    test_loss, test_auc, test_fmax,y_true, y_pred_sigm = evaluate(opt,device, net, criterion, test_loader, nth=10)

    # Save predictions
    if save_file is not None:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred_sigm}, open(save_file, 'wb'))
    # Display evaluation metrics
    print("--- test loss:                  %.4f" % test_loss)
    print("--- test roc auc score:         %.4f" % test_auc)
    print("--- test max F-score:           %.4f" % test_fmax)

    return epoch_num,test_loss, test_auc,test_fmax, y_true, y_pred_sigm

def evaluate(opt,device,net, criterion, eval_loader,nth=10):
    # Eval each sample

    net.eval()
    avg_loss = 0.0
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():   # set all 'requires_grad' to False
        pbar = tqdm(eval_loader)
        for data in pbar:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y
            # Forward pass
            outputs = net(data)
            current_loss = criterion(outputs, labels.float())
            avg_loss += current_loss.item() / len(eval_loader)

            y_true.append(labels.cpu().numpy().squeeze())
            y_pred_sigm.append(outputs.cpu().numpy().squeeze())
            pbar.set_postfix({'loss': current_loss.item()})
            pbar.update()

        # Calculate evaluation metrics
        y_true = np.vstack(y_true)
        y_pred_sigm = np.vstack(y_pred_sigm)

        roc_auc = compute_roc(y_true, y_pred_sigm)
        # # Maximum F-score
        avg_fmax = compute_fmax(y_true, y_pred_sigm, nrThresholds=nth)

    return avg_loss, roc_auc, avg_fmax,y_true, y_pred_sigm


def load_checkpoint(net, optimizer=None, scheduler=None, filename='model_best.pth.tar'):
    start_epoch = 0
    try:
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("\n[*] Loaded checkpoint at epoch %d" % start_epoch)
    except:
        print("[!] No checkpoint found, start epoch 0")

    return start_epoch

def plot_loss(n,save_dir):
    y = []
    enc = np.load(save_dir + "Epoch_{}.npy".format(n))
    tempy = list(enc)
    y += tempy

    x = range(0,len(y))
    plt.plot(x,y,",-")
    plt_title = 'BATCH_SIZE = 32; LEARNING_RATE:0.00005'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('LOSS')
    plt.savefig(save_dir + "Epoch_{}".format(n))

