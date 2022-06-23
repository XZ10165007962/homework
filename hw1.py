# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         hw1
# Description:
# Author:       xinzhuang
# Date:         2022/1/31
# 全连接神经网络做回归
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import csv

from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def feature_select(train):
    x = train[train.columns[1:94]]
    y = train[train.columns[94]]

    x = (x - x.min()) / (x.max() - x.min())

    bestfeature = SelectKBest(score_func=f_regression, k=5)
    fit = bestfeature.fit(x, y)
    dfsores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    featureScores = pd.concat([dfcolumns, dfsores], axis=1)
    featureScores.columns = ["Specs", "Score"]
    print(featureScores.nlargest(15, "Score"))
    return featureScores


class Covid19Sets(Dataset):
    def __init__(self, path, model='train'):
        super().__init__()
        self.model = model
        with open(path) as file:
            data_csv = list(csv.reader(file))
            data = np.array(data_csv[1:])[:, 1:].astype(float)
        if model == 'test':
            data = data[:, 0: 93]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, 0: 93]
            train_index = []
            dev_index = []
            for i in range(data.shape[0]):
                if i % 10 == 0:
                    dev_index.append(i)
                else:
                    train_index.append(i)
            if model == 'train':
                self.target = torch.FloatTensor(target[train_index])
                self.data = torch.FloatTensor(data[train_index, 0: 93])
            else:
                self.target = torch.FloatTensor(target[dev_index])
                self.data = torch.FloatTensor(data[dev_index, 0: 93])
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0)) / self.data[:, 40:].std(dim=0)
        self.dim = self.data.shape[1]

    def __getitem__(self, item):
        if self.model == 'train' or self.model == 'dev':
            return self.data[item], self.target[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0):
    dataset = Covid19Sets(path, model=mode)
    dataloder = DataLoader(
        dataset, batch_size,
        shuffle=(mode == "train"),
        drop_last=False,
        num_workers=n_jobs,
        pin_memory=False
    )
    return dataloder


class Mymodel(nn.Module):
    def __init__(self, input_dim):
        super(Mymodel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.critertion = nn.MSELoss(reduction="mean")

    def forward(self, input):
        return self.net(input).squeeze(1)

    def cal_loss(self, pred, target):
        return self.critertion(pred, target)


def train(model, train_data, dev_data):
    max_epoch = 1000
    epoch = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loss = []
    dev_loss = []
    break_flag = 0
    min_mse = 1000
    while epoch < max_epoch:
        model.train()
        for input, label in train_data:
            optimizer.zero_grad()
            output = model(input)
            loss = model.cal_loss(output, label)
            train_loss.append(loss.detach())
            loss.backward()
            optimizer.step()
        dev_mse = dev(model, dev_data)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Save model epoch = {:4d}, loss = {:.4f}'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), "/Users/xinzhuang/Documents/ml2021spring-hw1/mymodel.pth")
            break_flag = 0
        else:
            break_flag += 1
        dev_loss.append(dev_mse.detach())
        if break_flag > 200:
            break
        epoch += 1

    return train_loss, dev_loss


def dev(model, dev_data):
    model.eval()
    total_loss = []
    for input, label in dev_data:
        output = model(input)
        total_loss.append(model.cal_loss(output, label))
    return sum(total_loss) / len(total_loss)


def test(model, test_data):
    model.eval()
    output = []
    for input in test_data:
        pred = model(input)
        output.append(pred.detach())
    output = torch.cat(output, dim=0).numpy()
    return output


# 绘制学习曲线
def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(dev_loss)]
    plt.figure(1, figsize=(6, 4))
    plt.plot(x_1, train_loss, c='tab:red', label='train')
    plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    plt.figure(2, figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):  # enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            writer.writerow([i, p])


train_data = prep_dataloader("/Users/xinzhuang/Documents/ml2021spring-hw1/covid.train.csv", "train", batch_size=135)
dev_data = prep_dataloader("/Users/xinzhuang/Documents/ml2021spring-hw1/covid.train.csv", "dev", batch_size=135)
test_data = prep_dataloader("/Users/xinzhuang/Documents/ml2021spring-hw1/covid.test.csv", "test", batch_size=135)

myseed = 2020815
np.random.seed(myseed)
torch.manual_seed(myseed)

mymodel = Mymodel(train_data.dataset.dim)
train_loss, dev_loss = train(mymodel, train_data, dev_data)
plot_learning_curve(train_loss, dev_loss, title='deep model')
del mymodel

model = Mymodel(train_data.dataset.dim)
ckpt = torch.load('/Users/xinzhuang/Documents/ml2021spring-hw1/mymodel.pth', map_location='cpu')  # 加载最好的模型
model.load_state_dict(ckpt)
plot_pred(dev_data, model, 'cpu')
preds = test(model, test_data)

save_pred(preds, 'mypred.csv')
print('all done')