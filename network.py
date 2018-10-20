import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import load_data
import torch.optim as optim
import my_utils
import random
from torch.autograd import Variable


def pooling(x):
    # return my_utils.max_pooling(x)
    # return my_utils.mean_pooling(x)
    return my_utils.max_pooling(x)


class MI_Net_RS(nn.Module):
    def __init__(self,input):
        super(MI_Net_RS,self).__init__()
        self.fc_ = nn.Sequential(
            nn.Linear(input,128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,xx):
        xx_1 = self.fc_(xx)
        xx1 = F.dropout(xx_1)
        xx1 = pooling(xx1)

        xx_2 = self.fc(xx_1)
        xx2 = F.dropout(xx_2)
        xx2 = pooling(xx2)

        xx_3 = self.fc(xx_2)
        xx3 = F.dropout(xx_3)
        xx3 = pooling(xx3)

        xx_out = xx1+xx2+xx3 # connection

        return self.out(xx_out)

class MI_Net(nn.Module):
    def __init__(self,num_input):
        super(MI_Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_input, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc = nn.Linear(64,1)

    def forward(self, x):
        xx = self.net(x)
        xx = pooling(xx)
        xx = F.sigmoid(self.fc(xx))
        return xx

# max,lse,mean
class MI_Net_DS_te(nn.Module):
    def __init__(self,input):
        super(MI_Net_DS_te,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input,256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True)
        )
        self.fc_1 = nn.Linear(256,1)
        self.fc_2 = nn.Linear(128, 1)
        self.fc_3 = nn.Linear(64, 1)
    def forward(self, xx_,pooling_methods):
        x_1 = self.fc1(xx_)
        xx_1 = F.dropout(x_1)
        xx_1 = my_utils.pooling_m(xx_1,pooling_methods)

        xx_1 = F.sigmoid(self.fc_1(xx_1))

        x_2 = self.fc2(x_1)
        xx_2 = F.dropout(x_2)
        xx_2 = my_utils.pooling_m(xx_2,pooling_methods)
        xx_2 = F.sigmoid(self.fc_2(xx_2))

        x_3 = self.fc3(x_2)
        xx_3 = F.dropout(x_3)
        xx_3 = my_utils.pooling_m(xx_3,pooling_methods)
        xx_3 = F.sigmoid(self.fc_3(xx_3))

        xx = (xx_1+xx_2+xx_3)/3
        return xx


class MI_Net_DS(nn.Module):
    def __init__(self,input):
        super(MI_Net_DS,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input,256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True)
        )
        self.fc_1 = nn.Linear(256,1)
        self.fc_2 = nn.Linear(128, 1)
        self.fc_3 = nn.Linear(64, 1)
    def forward(self, xx_):
        x_1 = self.fc1(xx_)
        xx_1 = F.dropout(x_1)
        xx_1 = pooling(xx_1)

        xx_1 = F.sigmoid(self.fc_1(xx_1))

        x_2 = self.fc2(x_1)
        xx_2 = F.dropout(x_2)
        xx_2 = pooling(xx_2)
        xx_2 = F.sigmoid(self.fc_2(xx_2))

        x_3 = self.fc3(x_2)
        xx_3 = F.dropout(x_3)
        xx_3 = pooling(xx_3)
        xx_3 = F.sigmoid(self.fc_3(xx_3))

        xx = (xx_1+xx_2+xx_3)/3
        return xx


class mi_Net(nn.Module):
    def __init__(self,num_input):
        super(mi_Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_input, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        xx = self.net(x)
        xx = pooling(xx)
        return xx


def train(model, train_set, optimizer,epoch,device):
    # convert bag to batch
    random.shuffle(train_set)
    model.train()
    num_train_batch = len(train_set)
    train_loss = np.zeros((num_train_batch, 1), dtype=float)
    train_acc = np.zeros((num_train_batch, 1), dtype=float)
    for batch_idx,(data,target) in enumerate(train_set):
        optimizer.zero_grad()
        data = torch.Tensor(data).to(device)
        target = torch.Tensor(target).to(device)
        data,target = Variable(data),Variable(target)
        output = model(data)
        # print("output:",output)
        # print("target:",target)
        # loss = nn.CrossEntropyLoss()
        # loss_ = loss(target,output)
        loss = my_utils.my_loss(target,output)
        train_loss[batch_idx] = float(loss[0])
        train_acc[batch_idx] = np.floor(float(output[0][0])+0.5) == target[0]
        loss.backward()
        optimizer.step()
    return train_loss.mean(),train_acc.mean()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    random.shuffle(test_loader)
    for data, target in test_loader:
        data, target = torch.Tensor(data).to(device), torch.Tensor(target).to(device)
        data,target = Variable(data),Variable(target)
        # print("answers:", target)
        output = model(data)
        # print("prediction  ",output)
        test_loss += my_utils.my_loss(target, output)  # sum up batch loss
        if output[0][0]>0.5:
            output = 1
        else:
            output = 0
        correct +=(output==int(target[0]))
    test_loss /= len(test_loader)
    test_loss = test_loss[0]
    return test_loss,correct / len(test_loader)


if __name__ =="__main__":
    times = 5
    _epochs = 60
    n_folds = 10
    device = torch.device("cuda")
    # dataset_name_list = ["elephant_100x100_matlab","fox_100x100_matlab","tiger_100x100_matlab","musk1norm_matlab","musk2norm_matlab"]
    dataset_name_list = ["/home/zhangwenqiang/PycharmProjects/minn/data/comp.graphics.mat"]
    model_name = [MI_Net_DS]
    result_acc = np.zeros((len(model_name),len(dataset_name_list)))
    result_std = np.zeros((len(model_name),len(dataset_name_list)))
    for ca,_model in enumerate(model_name):
        print(_model.__name__, "\n", "-" * 40, "\n")
        for cb,dataset_name in enumerate(dataset_name_list):
            dataset_load = load_data.load_dataset(dataset_name,n_folds)
            temp_test_acc = np.zeros((times,n_folds))
            for time in range(times):
                for ifold in range(n_folds):
                    input_size = my_utils.decide_iput_size(dataset_name)
                    model = _model(input_size)
                    model.to(device)
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-4)
                    print("epoch ", time, " folds ", ifold)
                    train_bags = dataset_load[ifold]['train']
                    test_bags = dataset_load[ifold]['test']
                    train_set = my_utils.convertToBatch(train_bags)
                    test_set = my_utils.convertToBatch(test_bags)
                    for epoch in range(_epochs):
                        train_loss, train_acc = train(model, train_set, optimizer, epoch, device)
                        test_loss,test_acc = test(model,device,test_set)
                        print('epoch=', epoch, '  train_loss= {:.3f}'.format(train_loss), '  train_acc= {:.3f}'.format(
                            train_acc), '  test_loss={:.3f}'.format(test_loss), '  test_acc= {:.3f}'.format(test_acc))
                    temp_test_acc[time][ifold] = test_acc
                    print("test_acc:",test_acc)
            acc = temp_test_acc.mean()
            std = temp_test_acc.std()
            result_acc[ca][cb] = acc
            result_std[ca][cb] = std
            print(dataset_name,":",acc,"+",std)
    print(result_acc)
    print(result_std)
