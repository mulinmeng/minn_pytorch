import torch
import numpy as np
import torch

def max_pooling(x):
    return torch.max(x,dim=0,keepdim=True)[0]


def mean_pooling(x):
    return torch.mean(x,dim=0,keepdim=True)


def LSE_pooling(x):
    return torch.log(torch.mean(torch.exp(x), dim=0,keepdim=True))

def pooling_m(x,pooling_methods):
    if pooling_methods is 'max':
        return max_pooling(x)
    if pooling_methods is 'mean':
        return mean_pooling(x)
    else:
        return LSE_pooling(x)

def my_loss(y_true,y_pred):
    y_true = torch.mean(y_true,dim=0,keepdim=False)
    y_pred = torch.mean(y_pred,dim=0,keepdim=False)
    return -(1-y_true)*torch.log(1-y_pred+1e-6)-y_true*torch.log(y_pred+1e-6)


def convertToBatch(bags):
    """Convert to batch format.
    Parameters
    -----------------
    bags : list
        A list contains instance features of bags and bag labels.
    Return
    -----------------
    data_set : list
        Convert dataset to batch format(instance features, bag label).
    """
    batch_num = len(bags)
    data_set = []
    for ibag, bag in enumerate(bags):
        batch_data = np.asarray(bag[0], dtype='float32')*100
        batch_label = np.asarray(bag[1])
        data_set.append((batch_data, batch_label))
    return data_set


def decide_iput_size(name):
    if name.startswith("musk") is True:
        return 166
    else:
        if name.startswith("data") is True:
            return 200
    return 230