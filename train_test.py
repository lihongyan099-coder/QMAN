import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import d2l
import numpy as np
import time, os
import scipy.io as sio
import collections
import matplotlib.pyplot as plt
from operator import truediv
from sklearn import metrics
from util import sampling, get_device, generate_batch, print_results
from data import lazyprocessing
from torchsummary import summary
from thop import profile
from QMAN import*

device = get_device()   


def resolve_dict(hp):   
    return hp['dataset'], hp['run_times'], hp['pchannel'], hp['model'], hp['ws'], hp['epochs'], \
           hp['batch_size'], hp['learning_rate'], hp['train_proportion'], hp['train_num'], hp['outputmap'], \
           hp['only_draw_label'], hp['disjoint']


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=0)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def list_to_colormap(x_list):  
    color_map = {
        0: [0.5, 0.5, 0.5],  
        1: [1, 0, 0],  
        2: [0, 1, 0],  
        3: [0, 0, 1], 
        4: [1, 1, 0],  
        5: [1, 0, 1],  
        6: [0, 1, 1],  
        7: [0.5, 0, 0],
        8: [0, 0.5, 0],  
        9: [0.5, 0.5, 1],  
        10: [0.8, 0.8, 0],  
        11: [0.8, 0, 0.8], 
        12: [0, 0.8, 0.8],  
        13: [0.2, 0.2, 0.2],  
        14: [1, 0.647, 0], 
        15: [0.933, 0.863, 0.502],  
        16:[0,0,0]
    } 
    y = np.zeros((x_list.shape[0], 3))  
    for index, item in enumerate(x_list):  
        if item in color_map:  
            y[index] = np.array(color_map[item])  
        else:  
            y[index] = np.array([0, 0, 0])  
    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    dataset_name='BT'
    row = dataset_size_dict[dataset_name][0]
    col = dataset_size_dict[dataset_name][1]
    fig.set_size_inches( col* 2.0 / dpi,  row* 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0

def generate_png(all_loader, net, gt_hsi, Dataset, device ,model_save_path,total_indices,dataset_name='BT'):
    row = dataset_size_dict[dataset_name][0]
    col = dataset_size_dict[dataset_name][1]
    pred_test= []
    with torch.no_grad():
        for X, y in all_loader:
            X = torch.tensor(X).float().to(device)  
            y = torch.tensor(y).long().to(device)  
            y_hat = net(X)
            pred_test.extend(y_hat.argmax(dim=1).cpu().numpy())
    gt = gt_hsi.flatten()
    x_label = np.full(gt.shape, -1)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 0  
            x_label[i] = 16
    gt = gt[:] - 1
    if len(total_indices) != len(pred_test):
        print("Error: The lengths of total_indices and pred_test do not match.")
        return
    
    if max(total_indices) >= len(gt):
        print("Error: total_indices contains out-of-bounds indices.")
        return
    
    x_label[total_indices] = pred_test
    if np.any(x_label == -1):
        print('warning: there are unfilled elements in x_label')
        x_label[x_label == -1] = 0
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)#
    gt_re = np.reshape(y_gt, (row, col, 3))
    path = './' + net.name
    save_dir = os.path.join(path)  
    os.makedirs(save_dir, exist_ok=True)  
    
    classification_map(y_re, gt_hsi, 300,  
                       os.path.join(save_dir, f'{Dataset}_{net.name}.png'))   
    classification_map(gt_re, gt_hsi, 300,  
                       os.path.join(save_dir, f'{Dataset}_gt.png')) 
    
    print('------Get classification maps successful-------')  


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            n += y.shape[0]
    net.train()
    return [acc_sum / n, test_l_sum / test_num]
def save_model(net, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def validate_model(net, val_loader, loss, device):
    net.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X = torch.tensor(X).float().to(device)  
            y = torch.tensor(y).long().to(device)   
            y_hat = net(X)
            total_loss += loss(y_hat, y).item()
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
    return total_loss / (total_samples or 1), total_acc / (total_samples or 1)

def train(
    net, train_idx, val_idx, ground_truth, ground_test, ground_truth_val,
    X_PCAMirrow, batch_size, ws, dataset_name, loss, optimizer, device,
    epochs, 
    model_save_path="D:/pytest/my paper/second1/IN3/models/best_model.pt"
):
    net.to(device)
    best_val_acc, best_epoch= 0.0,-1
    train_loss_list, valida_loss_list = [], []
    train_acc_list, valida_acc_list = [], []
    
    train_time_list = []  
    val_time_list = []    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 2, gamma=0.5)
    for epoch in range(epochs):
        tic_train = time.time()
        net.train()
        train_loader = generate_batch(train_idx, X_PCAMirrow, ground_truth, batch_size, ws, dataset_name, shuffle=True)
        train_loss, train_acc, total_samples = 0.0, 0.0, 0
        for X, y in train_loader:
            X = torch.tensor(X).float().to(device)  
            y = torch.tensor(y).long().to(device)    
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

        train_loss /= (total_samples or 1)
        train_acc /= (total_samples or 1)
        toc_train = time.time()
        train_time = toc_train - tic_train
        train_time_list.append(train_time)
        tic_val = time.time()
        val_loader = generate_batch(val_idx, X_PCAMirrow, ground_truth_val, batch_size, ws, dataset_name, shuffle=False)
        valida_loss, valida_acc = validate_model(net, val_loader, loss, device)
        toc_val = time.time()  
        val_time = toc_val - tic_val
        val_time_list.append(val_time)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)


        if valida_acc > best_val_acc:
            best_val_acc = valida_acc
            best_epoch = epoch
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(net.state_dict(), model_save_path)

        print(f"Epoch {epoch + 1:03d}: train loss={train_loss:.5f}, train acc={train_acc:.2%}, train time={toc_train - tic_train:.2f}s, "
                f"val loss={valida_loss:.5f}, val acc={valida_acc:.2%}, val time={toc_val - tic_val:.2f}s")


    print(f"Training complete. Best model from epoch {best_epoch + 1} with validation accuracy: {best_val_acc:.2%}")
    
    return train_loss_list, train_acc_list, valida_loss_list, valida_acc_list,train_time_list, val_time_list

def test(net, test_loader,model_save_path, device):
    net.eval()
            
    if os.path.exists(model_save_path):
        net.load_state_dict(torch.load(model_save_path), strict=True)  
        print("Best model loaded successfully for testing.")
    else:
        raise FileNotFoundError(f"Best model file not found: {model_save_path}")
    
    pred_test, manifold = [], []
    tic_test = time.time()
    with torch.no_grad():
        for X, y in test_loader:
            X = torch.tensor(X).float().to(device)  
            y = torch.tensor(y).long().to(device)    
            y_hat = net(X)
            pred_test.extend(y_hat.argmax(dim=1).cpu().numpy())
            manifold.extend(y.cpu().numpy())
    toc_test = time.time()
    test_time = toc_test - tic_test
    return np.array(pred_test), np.array(manifold),test_time

def loop_train_test(hyper_parameter):
    datasetname, run_times, num_PC, model_type, ws, epochs, batch_size, lr, \
    train_proportion, num_list, outputmap, only_draw_label, disjoint = resolve_dict(hyper_parameter)

    print('>' * 10, "Data set Loading", '<' * 10)
    X_PCAMirrow, ground_truth, shapelist, hsidata = lazyprocessing(datasetname, num_PC=30, ws=ws, disjoint=disjoint)
    classnum = np.max(ground_truth)
    print(datasetname, 'shape:', shapelist)

    KAPPA, OA, AA, TRAINING_TIME, TESTING_TIME = [], [], [], [], []
    ELEMENT_ACC = np.zeros((run_times, classnum))
    all_train_times = np.zeros((run_times, epochs)) 
    all_val_times = np.zeros((run_times, epochs))   
    all_test_times = []                        
    
    print('>' * 10, "Start Training", '<' * 10)
    for run_i in range(run_times):
        print(f"Round {run_i + 1}/{run_times}")
        net = MultiBranchNet(input_channels=30, num_branches=3, out_features=classnum).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
        class_weights = torch.tensor([1.5,3.5, 2.0, 2.0, 1.5, 2.0, 2.0, 2.5, 1.5, 1.5, 1.0, 2.0,2.0, 3.5])#BT     
        loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        if disjoint:
            train_idx, test_idx, val_idx, *_ = sampling_disjoint(ground_truth)
            ground_truth_train, ground_truth_test = ground_truth[0], ground_truth[1]
        else:
            train_idx, test_idx, val_idx, total_indices, *_= sampling(ground_truth, train_proportion, num_list, seed=(run_i + 1) * 111)
            ground_truth_train, ground_truth_test, ground_truth_val = ground_truth, ground_truth, ground_truth

        print(f"Training samples: {len(train_idx)}, Testing samples: {len(test_idx)}, Validation samples: {len(val_idx)}")
        
        
        model_save_path = f"D:/pytest/my paper/second1/IN3/models/best_model.pt"
        tic_train = time.time()
        train_losses, train_accs, val_losses, val_accs,train_time_list, val_time_list = train(
            net, train_idx, val_idx, ground_truth_train, ground_truth_test, ground_truth_val,
            X_PCAMirrow, batch_size, ws, datasetname, loss, optimizer, device, epochs
        )
        toc_train = time.time()
        all_train_times[run_i] = train_time_list
        all_val_times[run_i] = val_time_list


        test_loader = generate_batch(test_idx, X_PCAMirrow, ground_truth_test, batch_size * 3, ws, datasetname, shuffle=False)
        net.to(device)  
        tic_test = time.time()
        pred_test, manifold,test_time = test(net, test_loader,model_save_path, device)
        all_test_times.append(test_time)
        toc_test = time.time()
        gt_test = manifold
        overall_acc = metrics.accuracy_score(pred_test, gt_test)
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
     
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test)

        print(confusion_matrix)
        print(f"OA: {overall_acc:.2%}, AA: {average_acc:.2%}, Kappa: {kappa:.4f}")
        print(f"Each acc: {each_acc}")
        print(f"Training time: {toc_train - tic_train:.2f}s, Testing time: {toc_test - tic_test:.2f}s")
  
        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(toc_train - tic_train)
        TESTING_TIME.append(toc_test - tic_test)
        ELEMENT_ACC[run_i, :] = each_acc
        
    avg_train_times = np.mean(all_train_times, axis=0)
    avg_val_times = np.mean(all_val_times, axis=0)
    avg_test_time = np.mean(all_test_times)

    print('Training and testing completed.')
    print_results(datasetname, np.array(OA), np.array(AA), np.array(KAPPA), np.array(ELEMENT_ACC),
                np.array(TRAINING_TIME), np.array(TESTING_TIME))

    all_loader = generate_batch(total_indices, X_PCAMirrow, ground_truth, batch_size * 3, ws, datasetname, shuffle=False)
    #generate_png(all_loader, net, ground_truth, datasetname, device,model_save_path, total_indices,dataset_name='PU')       
    
    return {
        "KAPPA": KAPPA,
        "OA": OA,
        "AA": AA,
        "TRAINING_TIME": TRAINING_TIME,
        "TESTING_TIME": TESTING_TIME,
        "ELEMENT_ACC": ELEMENT_ACC
    }
