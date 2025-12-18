import torch
import numpy as np
import os
import sklearn
import scipy.io as sio
from scipy.io import savemat, loadmat
from data import*
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from scipy.io import savemat
import os



def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def print_results(data_name, oa, aa, kappa, class_acc, traintime, testtime):
    log_file = f"{data_name}_results.txt"
    with open(log_file, "a") as f:
        n_class = dataset_class_dict[data_name]
        mean_oa = format(np.mean(oa * 100), '.2f')
        std_oa = format(np.std(oa * 100), '.2f')
        mean_aa = format(np.mean(aa) * 100, '.2f')
        std_aa = format(np.std(aa) * 100, '.2f')
        mean_kappa = format(np.mean(kappa) * 100, '.2f')
        std_kappa = format(np.std(kappa) * 100, '.2f')

        train_time_str = f"Train time: {np.mean(traintime):.2f} ± {np.std(traintime):.2f}"
        test_time_str = f"Test time: {np.mean(testtime):.2f} ± {np.std(testtime):.2f}"
        print(train_time_str)
        print(test_time_str)
        f.write(train_time_str + "\n")
        f.write(test_time_str + "\n")
        
        for i in range(n_class):
            mean_std = f"{np.mean(class_acc[:, i]) * 100:.2f} ± {np.std(class_acc[:, i]) * 100:.2f}"
            class_acc_str = f"Class {i + 1} mean ± std: {mean_std}"
            print(class_acc_str)
            f.write(class_acc_str + "\n")
        
        oa_str = f"OA mean: {mean_oa} ± {std_oa}"
        aa_str = f"AA mean: {mean_aa} ± {std_aa}"
        kappa_str = f"Kappa mean: {mean_kappa} ± {std_kappa}"
        
        print(oa_str)
        print(aa_str)
        print(kappa_str)
        f.write(oa_str + "\n")
        f.write(aa_str + "\n")
        f.write(kappa_str + "\n")
        f.write("\n")  


def sampling(ground_truth, train_proportion=0.2, val_proportion=0.1, train_list=[], seed=666):
    if not isinstance(val_proportion, (int, float)):
        if isinstance(val_proportion, list) and not val_proportion:
            val_proportion = 0.1  
        else:
            raise ValueError(f"val_proportion should be a number, got {val_proportion} of type {type(val_proportion)}")
    random_state = np.random.RandomState(seed=seed)
    train = {}
    val = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)[0]
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        random_state.shuffle(indexes)
        labels_loc[i] = indexes
        nb_train = max(int(train_proportion * len(indexes)), 5)
        nb_val = max(int(val_proportion * len(indexes)), 5)

        train[i] = indexes[:nb_train]
        val[i] = indexes[nb_train:nb_train + nb_val]
        test[i] = indexes[nb_train + nb_val:]

    train_indexes = []
    val_indexes = []
    test_indexes = []

    for i in range(m):
        train_indexes += train[i]
        val_indexes += val[i]
        test_indexes += test[i]
        
    total_indices = np.array(train_indexes + val_indexes + test_indexes)

    random_state.shuffle(train_indexes)
    random_state.shuffle(val_indexes)
    random_state.shuffle(test_indexes)
    random_state.shuffle(total_indices)

    train_idx = np.array(train_indexes)
    val_idx = np.array(val_indexes)
    test_idx = np.array(test_indexes)

    drawlabel_idx = np.array(train_indexes + test_indexes)
    drawall_idx = np.array([j for j, x in enumerate(ground_truth.ravel().tolist())])

    # # 保存索引到 .MAT 文件
    # output_dir = './output'
    # os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    # savemat(os.path.join(output_dir, 'indexes217.mat'), {
    #     'train_indexes': train_idx,
    #     'val_indexes': val_idx,
    #     'test_indexes': test_idx,
    # })

    # print(f"索引文件已保存为 .MAT 格式到 {output_dir} 目录下")
    return train_idx, test_idx, val_idx, total_indices, drawlabel_idx, drawall_idx


                                                                

def sampling_disjoint(ground_truth):
    Y_train = ground_truth[0]
    Y_test = ground_truth[1]
    n_class = Y_test.max()
    train_idx = list()
    test_idx = list()
    val_idx = list()
    for i in range(1, n_class + 1):
        train_i = np.where(Y_train == i)[0]
        test_i = np.where(Y_test == i)[0]
        
        train_idx.extend(train_i[:int(len(train_i)*1)])
        val_idx.extend(train_i[int(len(train_i)*1):])
        test_idx.extend(test_i)
    
    drawlabel_idx = np.array(train_idx + val_idx + test_idx)
    drawall_idx = np.array([j for j, x in enumerate(Y_train)])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    return train_idx, test_idx, val_idx, drawlabel_idx, drawall_idx



def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)  
    x_train_band = np.zeros_like(x_train_reshape, dtype=float)
    x_train_band[:] = x_train_reshape[:]
    for i in range(nn):
        x_train_band[:, :, :i + 1] += x_train_reshape[:, :, band - i - 1:]
        x_train_band[:, :, i + 1:] += x_train_reshape[:, :, :band - i - 1]
    for i in range(nn):
        x_train_band[:, :, band - i - 1:] += x_train_reshape[:, :, :i + 1]
        x_train_band[:, :, :band - i - 1] += x_train_reshape[:, :, i + 1:]
    return x_train_band.reshape(x_train.shape)


def generate_batch(idx, X_PCAMirrow, Y, batch_size, ws, dataset_name, shuffle=False):
    num = len(idx)
    hw = ws // 2
    X_PCAMirrow = PCAMirrowCut(dataset_name, X_PCAMirrow, hw, num_PC=30, dr_flag=True)
    row = dataset_size_dict[dataset_name][0]
    col = dataset_size_dict[dataset_name][1]
    
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, num, batch_size):
        bi = np.array(idx)[np.arange(i, min(num, i + batch_size))]
        index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)
        index_col = (bi + 1) - (index_row - 1) * col
        patches = np.zeros([bi.size, ws, ws, X_PCAMirrow.shape[-1]])
        for j in range(bi.size):
            a = index_row[j] - 1
            b = index_col[j] - 1
            patch = X_PCAMirrow[a:a + ws, b:b + ws, :]  
            patches[j, :, :, :] = patch 
        labels = Y[bi, :]-1 
        patches=gain_neighborhood_band(x_train=patches,band=X_PCAMirrow.shape[2],band_patch=3,patch=ws)
        yield patches, labels[:,0] 


