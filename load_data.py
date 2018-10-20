import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold


def load_dataset(dataset_nm, n_folds):
    """Load data from file, do pre-processing, split it into train/test set.
    Parameters
    -----------------
    dataset_nm : string
        Name of dataset.
    n_folds : int
        Number of cross-validation folds.
    Returns
    -----------------
    datasets : list
        List contains split datasets for k-Fold cross-validation.
    """
    # load data from file
    data = sio.loadmat('./dataset/'+dataset_nm+'.mat')
    print(data)
    temp = np.ones((1,data['labels'].shape[1]))
    ins_fea = data['features']
    ins_fea = np.array(np.zeros(ins_fea.shape)+ins_fea)

    bags_label = np.array((data['labels']+temp)//2,dtype=int)
    bag_ids = data['bag_ids'][0]

    # store data in bag level
    ins_idx_of_input = {}
    for id, bag_nm in enumerate(bag_ids):
        if bag_nm in ins_idx_of_input.keys():
            ins_idx_of_input[bag_nm].append(id)
        else:
            ins_idx_of_input[bag_nm] = [id]

    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        a=[]
        b=[]
        for ins_idx in ins_idxs:
            a.append(ins_fea[ins_idx])
            b.append(bags_label[0,ins_idx])
        bags_fea.append((a,b))
    num_bag = len(bags_fea)
    kf = KFold(n_splits=n_folds,shuffle=True,random_state=None)
    datasets = []
    for train_index,test_index in kf.split(bags_fea):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_index]
        dataset['test'] = [bags_fea[ibag] for ibag in test_index]
        datasets.append(dataset)
    return datasets


def load_news(dir_nm, n_folds):
    data = sio.loadmat(dir_nm)['data']
    # print(data)
    # print(data[0][0].shape)
    num_bags = data.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
    datasets = []
    for train_index,test_index in kf.split(data):
        dataset = {}
        dataset['train'] = [data[ibag] for ibag in train_index]
        dataset['test'] = [data[ibag] for ibag in test_index]
        datasets.append(dataset)
    return datasets


if __name__ =="__main__":
    # elephant_100x100_matlab
    # fox_100x100_matlab
    # musk1norm_matlab
    # musk2norm_matlab
    load_news("data/alt.atheism.mat",5)
    # load_dataset("20NewsGroups test, train",5)