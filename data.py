import scipy.io as sio 
import tifffile as tiff
import numpy as np 
import cv2, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from sklearn.decomposition import FastICA, PCA 
from sklearn import preprocessing
import sklearn.manifold as manifold
from pylab import *
from osgeo import gdal

mpl.rcParams['font.sans-serif'] = ['Arial']

data_name_dict = {
    'PU': 'PaviaU',
    'IN': 'Indian_pines',
    'HU': 'Houston',
    'SV': 'Salinas',
    'KSC': 'KSC',
    'BT': 'Botswana',
    'HY': 'HyRANK_satellite',
    'HU2018': 'Houston2018',
    'HanChuan': 'HanChuan',
    'HongHu': 'HongHu',
    'LongKou': 'LongKou',
    'ZY': 'ZY_hhk',
    'XA': 'XiongAn'
}

dataset_path = {
    'PU': ['D:/pytest/my paper/first3/PU/PU/datasets/PU/PaviaU.mat',
    #'PU':['D:/pytest/my paper/first3/PU/PU/PaviaU_filtered.mat',
           'D:/pytest/my paper/first3/PU/PU/datasets/PU/PaviaU_gt.mat'],
    'IN': ['D:/pytest/my paper/second1/IN3/datasets/IN/Indian_pines_corrected.mat',
           'D:/pytest/my paper/second1/IN3/datasets/IN/Indian_pines_gt.mat',
           './datasets/IN/train_gt.mat',
           './datasets/IN/test_gt.mat'],
    'SV': ['D:/pytest/my paper/first3/SA/pca=30_main_B=branch=3,band_G=3_train=20_layer=3_study_patch/datasets/SA//Salinas_corrected.mat',
           'D:/pytest/my paper/first3/SA/pca=30_main_B=branch=3,band_G=3_train=20_layer=3_study_patch/datasets/SA//Salinas_gt.mat'],
    'KSC': ['./datasets/KSC/KSC.mat',
            './datasets/KSC/KSC_gt.mat'],
    'HU': ['D:/pytest/my paper/second1/HU/datasets/HU/Houston.mat',
           'D:/pytest/my paper/second1/HU/datasets/HU/Houston_gt.mat',
           './datasets/HU/Houston_train_gt.mat',
           './datasets/HU/Houston_test_gt.mat'],
    'BT': ['D:/pytest/my paper/second1/IN3/BT/Botswana.mat',
           'D:/pytest/my paper/second1/IN3/BT/Botswana_gt.mat'],
    'HU2018': ['./datasets/HU2018/Houston2018.mat',
               './datasets/HU2018/Houston2018_gt.mat'],
    'HanChuan': ['./datasets/WH/HanChuan/HanChuan.mat',
                 './datasets/WH/HanChuan/HanChuan_gt.mat'],
    'HongHu': ['./datasets/WH/HongHu/HongHu.mat',
               './datasets/WH/HongHu/HongHu_gt.mat'],
    'LongKou': ['./datasets/WH/LongKou/LongKou.mat',
                './datasets/WH/LongKou/LongKou_gt.mat'],
    'ZY': ['./datasets/ZY_hhk/ZY_hhk.mat',
           './datasets/ZY_hhk/ZY_hhk_gt.mat'],
    'XA': ['./datasets/XA/XiongAn.mat',
           './datasets/XA/XiongAn_gt.mat'],
    
}

dataset_size_dict = {
    'PU': [610, 340, 103],
    'IN': [145, 145, 200],
    'HU': [349, 1905, 144],
    'KSC': [512, 614, 176],
    'SV': [512, 217, 204],
    'BT': [1476, 256, 145],
    'HU2018': [4172, 1202, 48],
    'HongHu': [940, 475, 270],
    'ZY': [1147, 1600, 119],
    'XA': [1580, 3750, 256],
}

dataset_class_dict = {
    'PU': 9,
    'IN': 16,
    'HU': 15,
    'KSC': 13,
    'SV': 16,
    'BT': 14,
    'HU2018': 20,
    'HongHu': 22,
    'ZY': 23,
    'XA': 20,

}
simple_num_dict = {
    'PU': 42776,
    'IN': 10249,
    'HU': 15029,
    'KSC': 5211,
    'SV': 54129,
    'BT': 3248,
    'HU2018': 2018910,
    'ZY': 9825,
    'XA': 3677110,
}
class_name_dict = {
    'PU': ["Asphalt", "Meadows", "Gravel", "Trees",
           "Painted metal sheets", "Bare Soil", "Bitumen",
           "Self-Blocking Bricks", "Shadows"],
    'SV': ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
           "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
           "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
           "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"],
    'IN': ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
           "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
           "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"],
    'BT': ["Water", "Hippo grass", "Floodplain grasses 1", "Floodplain grasses 2", "Reeds", "Riparian",
           "Firescar", "Island interior", "Acacia woodlands", "Acacia shrublands", "Acacia grasslands",
           "Short mopane", "Mixed mopane", "Exposed soils"],
    'KSC': ["Scrub", "Willow swamp", "Cabbage palm hammock", "Cabbage palm/oak hammock",
            "Slash pine", "Oak/broadleaf hammock", "Hardwood swamp", "Graminoid marsh",
            "Spartina marsh", "Cattail marsh", "Salt marsh", "Mud flats", "Wate"],
    'HU': ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
           'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
           'Railway', 'Parking Lot1', 'Parking Lot2', 'Tennis court', 'Running track'],
    'HU2018': ['Healthy Grass', 'Stressed Grass', 'Artificial Turf', 'Evergreen Trees',
               'Deciduous Trees', 'Bare Earth', 'Water', 'Residential Buildings',
               'Non-Residential Buildings', 'Roads', 'Sidewalks', 'Crosswalks',
               'Major Thoroughfares', 'Highways', 'Railways', 'Paved Parking Lots',
               'Unpaved Parking Lots', 'Cars', 'Trains', 'Stadium Seats'],
    'XA': ['Acer negundo Linn', 'Willow', 'Elm', 'Paddy', 'Chinese Pagoda Tree',
           'Fraxinus chinensis', 'Koelreuteria paniculata', 'Water', 'Bare land',
           'Paddy stubble', 'Robinia pseudoacacia', 'Corn', 'Pear', 'Soya', 'Alamo', 'Vegetable field',
           'Sparsewood', 'Meadow', 'Peach', 'Building'],
    'ZY': ['Reed', 'Spartina alterniflora', 'Salt filter pond', 'Salt evaporation pond',
           'Dry pond', 'Tamarisk', 'Salt pan', 'Seepweed', 'River', 'Sea', 'Mudbank', 'Tidal creek',
           'Fallow land', 'Ecological restoration pond', 'Robinia', 'Fishpond', 'Pit pond',
           'Building', 'Bare land', 'Paddyfield', 'Cotton', 'Soybean', 'Corn']
}


color_map_dict = {
    'PU': np.array([[0, 0, 255],
                    [76, 230, 0],
                    [255, 190, 232],
                    [255, 0, 0],
                    [156, 156, 156],
                    [255, 255, 115],
                    [0, 255, 197],
                    [132, 0, 168],
                    [0, 0, 0]]),
    'IN': np.array([[0, 168, 132],
                    [76, 0, 115],
                    [0, 0, 0],
                    [190, 255, 232],
                    [255, 0, 0],
                    [115, 0, 0],
                    [205, 205, 102],
                    [137, 90, 68],
                    [215, 158, 158],
                    [255, 115, 223],
                    [0, 0, 255],
                    [156, 156, 156],
                    [115, 223, 255],
                    [0, 255, 0],
                    [255, 255, 0],
                    [255, 170, 0]]),
    'HU': np.array([[0, 168, 132],
                    [76, 0, 115],
                    [0, 0, 0],
                    [190, 255, 232],
                    [255, 0, 0],
                    [115, 0, 0],
                    [205, 205, 102],
                    [137, 90, 68],
                    [215, 158, 158],
                    [255, 115, 223],
                    [0, 0, 255],
                    [156, 156, 156],
                    [115, 223, 255],
                    [0, 255, 0],
                    [255, 255, 0]]),
    'KSC': np.array([[0, 168, 132],
                     [76, 0, 115],
                     [255, 0, 0],
                     [190, 255, 232],
                     [0, 0, 0],
                     [115, 0, 0],
                     [205, 205, 102],
                     [137, 90, 68],
                     [215, 158, 158],
                     [255, 115, 223],
                     [0, 0, 255],
                     [156, 156, 156],
                     [115, 223, 255]]),
    'SV': np.array([[0, 168, 132],
                    [76, 0, 115],
                    [0, 0, 0],
                    [190, 255, 232],
                    [255, 0, 0],
                    [115, 0, 0],
                    [205, 205, 102],
                    [137, 90, 68],
                    [215, 158, 158],
                    [255, 115, 223],
                    [0, 0, 255],
                    [156, 156, 156],
                    [115, 223, 255],
                    [0, 255, 0],
                    [255, 255, 0],
                    [255, 170, 0]]),
    'BT': np.array([[0, 168, 132],
                    [76, 0, 115],
                    [0, 0, 0],
                    [190, 255, 232],
                    [255, 0, 0],
                    [115, 0, 0],
                    [205, 205, 102],
                    [137, 90, 68],
                    [215, 158, 158],
                    [255, 115, 223],
                    [0, 0, 255],
                    [156, 156, 156],
                    [115, 223, 255],
                    [0, 255, 0]]),
    
    'HU2018': np.array([[0, 205, 0],
                        [127, 255, 0],
                        [46, 139, 87],
                        [0, 139, 0],
                        [0, 70, 0],
                        [160, 82, 45],
                        [0, 255, 255],
                        [255, 255, 255],
                        [216, 191, 216],
                        [255, 0, 0],
                        [170, 160, 150],
                        [128, 128, 128],
                        [160, 0, 0],
                        [80, 0, 0],
                        [232, 161, 24],
                        [255, 255, 0],
                        [238, 154, 0],
                        [255, 0, 255],
                        [0, 0, 255],
                        [176, 196, 222]]),
    'XA': np.array([[0, 139, 0],
                    [0, 0, 255],
                    [255, 255, 0],
                    [0, 255, 0],
                    [255, 0, 255],
                    [139, 139, 0],
                    [0, 139, 139],
                    [0, 255, 255],
                    [0, 0, 139],
                    [255, 127, 80],
                    [127, 255, 0],
                    [218, 112, 214],
                    [46, 139, 87],
                    [0, 30, 190],
                    [255, 165, 0],
                    [127, 255, 212],
                    [218, 112, 214],
                    [255, 0, 0],
                    [205, 0, 0],
                    [139, 0, 0]]),
    'ZY': np.array([[0, 139, 0],
                    [0, 0, 255],
                    [255, 255, 0],
                    [255, 127, 80],
                    [255, 0, 255],
                    [139, 139, 0],
                    [0, 139, 139],
                    [0, 255, 0],
                    [0, 255, 255],
                    [0, 30, 190],
                    [127, 255, 0],
                    [218, 112, 214],
                    [46, 139, 87],
                    [0, 0, 139],
                    [255, 165, 0],
                    [127, 255, 212],
                    [218, 112, 214],
                    [255, 0, 0],
                    [205, 0, 0],
                    [139, 0, 0],
                    [65, 105, 225],
                    [240, 230, 140],
                    [244, 164, 96]]),
    'Hyrank': np.array([[0, 168, 132],
                    [76, 0, 115],
                    [0, 0, 0],
                    [190, 255, 232],
                    [255, 0, 0],
                    [115, 0, 0],
                    [205, 205, 102],
                    [137, 90, 68],
                    [215, 158, 158],
                    [255, 115, 223],
                    [0, 0, 255],
                    [156, 156, 156],
                    [115, 223, 255],
                    [0, 255, 0],
                    [255, 255, 0]]),
}

default_max_hw = 35


fc_dict = {
    'PU': [56, 33, 13, 1.0, 0.5],
    '2': [50, 27, 17, 1.0, 1.0],
    'HU': [59, 40, 23, 6.8, 0.65],
    '4': [59, 40, 23, 1.0, 0.7],
    'XA': [120, 72, 36, 1.0, 1],
    'ZY': [55, 28, 8, 1, 1],
}


def mat2rgb(mat, eps=0.0):
    sz = np.shape(mat)
    if len(sz) == 3:
        r = np.reshape(mat[:, :, 0], [sz[0] * sz[1]])
        r = np.expand_dims(np.reshape(featureNormalize(r, type=2, eps=eps), [sz[0], sz[1]]), axis=-1)
        g = np.reshape(mat[:, :, 1], [sz[0] * sz[1]])
        g = np.expand_dims(np.reshape(featureNormalize(g, type=2, eps=eps), [sz[0], sz[1]]), axis=-1)
        b = np.reshape(mat[:, :, 2], [sz[0] * sz[1]])
        b = np.expand_dims(np.reshape(featureNormalize(b, type=2, eps=eps), [sz[0], sz[1]]), axis=-1)
        rgb = np.concatenate([r, g, b], axis=-1)
        return rgb
    
    else:
        gray = np.reshape(mat[:, :], [sz[0] * sz[1]])
        gray = np.reshape(featureNormalize(gray, type=2, eps=eps), [sz[0], sz[1]])
        return gray


def draw_false_color(data_name='BT'):
    [x, _] = load_dataset(data_name)
    rgb = fc_dict[data_name]
    x = x[:, :, rgb[0:3]]
    
    plt.imsave('./datamap/' + data_name_dict[data_name] + '_1.png', x)
    plt.imshow(x)
    plt.axis('off')
    plt.show()



def load_dataset(dataset_name='HU', disjoint=False):
    if dataset_name not in dataset_path.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    path = dataset_path[str(dataset_name)]
    dataset_mat = sio.loadmat(path[0])
    key = [k for k in dataset_mat if not k.endswith('_')]
    hsidata = dataset_mat[key[0]]
    if disjoint:
        gt_mat_train = sio.loadmat(path[2])
        key_train = [k for k in gt_mat_train if not k.endswith('_')]
        gt_mat_test = sio.loadmat(path[3])
        key_test = [k for k in gt_mat_test if not k.endswith('_')]
        gt_train = gt_mat_train[key_train[0]]
        gt_test = gt_mat_test[key_test[0]]
        gt = [gt_train, gt_test]
    else:
        gt_mat = sio.loadmat(path[1])
        key = [k for k in gt_mat if not k.endswith('_')]
        gt = gt_mat[key[0]]
    return hsidata, gt


def featureNormalize(X, type, eps=0.0):
    if type == 1:
        mu = np.mean(X, 0)
        X_norm = X - mu
        sigma = np.std(X_norm, 0)
        X_norm = X_norm / sigma
        m = np.diag(1 / sigma)
        # return X_norm
        return X_norm, m
    elif type == 2:
        minX = np.min(X, 0)
        maxX = np.max(X, 0)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX + eps)
        return X_norm
    elif type == 3:
        sigma = np.std(X, 0)
        X_norm = X / sigma
        return X_norm


def dimensionReduction2d(x, num=3, type='pca'):
    def ica(x, num_CA):
        ica = FastICA(n_components=num_CA)
        c = ica.fit_transform(x)
        return c
    
    def pca(x, n_components=3):
        c = PCANorm(x, n_components) 
        return c
    
    def PCANorm(x, num_PC):
        mu = np.mean(x, 0)
        x_norm = x - mu
        sigma = np.cov(x_norm.T)
        [U, S, V] = np.linalg.svd(sigma)
        u = U[:, 0:num_PC]
        XPCANorm = np.dot(x_norm, u)
        return XPCANorm.astype(dtype=np.float32), x_norm.astype(dtype=np.float32), u
    
    if type == 'pca':
        c = pca(x, num)
    elif type == 'ica':
        c = ica(x, num)
    else:
        c = x.astype(dtype=np.float32)
    return c


def mirror_concatenate(x, max_hw=default_max_hw):
    x_extension = cv2.copyMakeBorder(x, max_hw, max_hw, max_hw, max_hw, cv2.BORDER_REFLECT)
    return x_extension


def PCAMirrowCut(dataset_name, X, hw, num_PC=30, dr_flag=True):
    cnum = dataset_size_dict[str(dataset_name)][2]
    [row, col, n_feature] = X.shape
    X = X.reshape(row * col, n_feature)
    if num_PC == n_feature:
        dr_flag = False
    else:
        dr_flag = True
    if dr_flag:
        X, X_norm, U = dimensionReduction2d(X, cnum)
        X= featureNormalize(X, type=2)
        X = X.reshape(row, col, X.shape[-1])
        print("PCA has done!")
    else:
        X = preprocessing.scale(X)
        
        X = X.reshape(row, col, X.shape[-1])
    X_extension = mirror_concatenate(X)
    
    
    if num_PC != 0:
        X_extension = X_extension[:, :, 0:num_PC]

    b = default_max_hw - hw
    X_extension = X_extension[b:-b, b:-b, :]
    return X_extension


def lazyprocessing(dataset_name, num_PC, ws=21, disjoint=False, dr_flag=False):
    hw = ws // 2
    X, Y = load_dataset(dataset_name, disjoint)
    [row, col, ch] = X.shape
    X_PCAMirrow = PCAMirrowCut(dataset_name, X, hw=hw, num_PC=num_PC, dr_flag=dr_flag)
    if disjoint:
        Y1 = Y[0]
        Y1 = Y1.reshape(row * col, 1)
        Y2 = Y[1]
        Y2 = Y2.reshape(row * col, 1)
        Y = [Y1, Y2]
    else:
        Y = Y.reshape(row * col, 1)
    return X_PCAMirrow, Y, [row, col, ch], X

