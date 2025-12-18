from train_test import loop_train_test
from data import dataset_size_dict
import warnings
import time
import torch

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
run_times = 1
output_map = False 
only_draw_label = False
disjoint = False 


model_type = 'QMAN' 
ws = 21
epochs = 300
batch_size = 64
lr = 0.001

def IN_experiment(): 
    train_proportion = 0.2
    val_proportion = 0.1
    
    num_list = []
    pcadimension = 0

    hp = {
        'dataset': 'BT',
        'run_times': run_times,
        'pchannel': dataset_size_dict['BT'][2] if pcadimension == 0 else pcadimension,
        'model': model_type,
        'ws': ws,
        'epochs': epochs ,
        'batch_size': batch_size,
        'learning_rate': lr,
        'train_proportion': train_proportion, 
        'val_proportion':val_proportion,
        'train_num': num_list,
        'outputmap': output_map,
        'only_draw_label': only_draw_label,
        'disjoint': False
    }
    loop_train_test(hp)


if __name__ == '__main__':
        IN_experiment()
        print(time.asctime(time.localtime()))