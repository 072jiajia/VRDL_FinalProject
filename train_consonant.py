import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch.nn as nn
import pandas as pd
import argparse

from model import APINet
from Dataset import GraphemeDataset, BalancedBatchSampler


os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


def train_one_epoch(model, train_loader, optimizer):
    ''' train 1 epoch '''
    model.train()
    running_loss = 0
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        # in training phase, model outputs the loss directly
        total_loss = model(inputs.unsqueeze(1).float(), labels)
        total_loss = total_loss.mean()
        total_loss.backward()

        running_loss += total_loss
        optimizer.step()

        # print loss of this batch and average loss of this epoch
        print(idx + 1, '/', len(train_loader), ' // ',
              float(total_loss), '(', float(running_loss / (idx + 1)), ')',
              end='           \r')

    # print result of this epoch
    print(' ' * 150, end='\r')
    print('loss : {:.4f}'.format(running_loss/len(train_loader)))


def validate(model, test_loader):
    ''' Do Validation '''
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            # make prediction
            outputs = model(inputs.unsqueeze(1).float())
            outputs = outputs.cuda()

            # compute loss and accuracy
            loss = criterion(outputs, labels)
            running_loss += loss
            running_acc += (outputs.argmax(1) == labels).float().mean()

        # print average loss and accuracy
        acc = running_acc*100/(len(test_loader))
        print('val_acc : {:.4f}%'.format(acc))
        print('va_loss : {:.4f}'.format(running_loss/len(test_loader)))
        return acc


def _init_():
    parser = argparse.ArgumentParser(description='VRDL Final_Project')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--n_classes', default=30, type=int,
                        help='number of classes inone training batch')
    parser.add_argument('--n_samples', default=4, type=int,
                        help='number of samples in each class')
    parser.add_argument('--KFold', default=20, type=int,
                        help='train K models for K-Fold')
    args = parser.parse_args()

    np.random.seed(0)

    return args


def get_indices(train):
    OK = {13, 22, 23, 29, 43, 53, 64, 71, 72,
          79, 96, 103, 107, 113, 147, 149, 159}
    Vowel = train.grapheme_root.values
    indices = []
    for i in range(len(Vowel)):
        if Vowel[i] in OK:
            indices.append(i)

    return indices


def main(args):
    # load data
    train = pd.read_csv('train.csv')

    # read files separately
    print('loading train_image_data_0.parquet')
    data0 = pd.read_parquet('train_image_data_0.parquet')
    train0 = train.loc[range(0, 50210)].reset_index(drop=True)
    indices = get_indices(train0)
    train0 = train0.loc[indices]
    data0 = data0.loc[indices]

    print('loading train_image_data_1.parquet')
    data1 = pd.read_parquet('train_image_data_1.parquet')
    train1 = train.loc[range(50210, 50210*2)].reset_index(drop=True)
    indices = get_indices(train1)
    train1 = train1.loc[indices]
    data1 = data1.loc[indices]

    print('loading train_image_data_2.parquet')
    data2 = pd.read_parquet('train_image_data_2.parquet')
    train2 = train.loc[range(50210*2, 50210*3)].reset_index(drop=True)
    indices = get_indices(train2)
    train2 = train2.loc[indices]
    data2 = data2.loc[indices]

    print('loading train_image_data_3.parquet')
    data3 = pd.read_parquet('train_image_data_3.parquet')
    train3 = train.loc[range(50210*3, 50210*4)].reset_index(drop=True)
    indices = get_indices(train3)
    train3 = train3.loc[indices]
    data3 = data3.loc[indices]

    # concat data
    data_full = pd.concat([data0, data1, data2, data3], ignore_index=True)
    train = pd.concat([train0, train1, train2, train3], ignore_index=True)

    # delete unused variables
    del data0
    del data1
    del data2
    del data3

    print("read finished")
    print('number of data:', len(data_full))

    # split data to training set and validation set
    L = len(data_full)
    perm = np.random.permutation(L)

    train_index = perm[:(args.KFold - 1) * L // args.KFold]
    test_index = perm[(args.KFold - 1) * L // args.KFold:]

    reduced_train = train.loc[train_index]
    train_data = data_full.loc[train_index]
    reduced_test = train.loc[test_index]
    test_data = data_full.loc[test_index]

    # get training set loader and test set loader
    train_image = GraphemeDataset(train_data,
                                  reduced_train.consonant_diacritic.values,
                                  'train')
    train_sampler = BalancedBatchSampler(train_image,
                                         n_classes=args.n_classes,
                                         n_samples=args.n_samples)
    train_loader = torch.utils.data.DataLoader(train_image,
                                               batch_sampler=train_sampler,
                                               num_workers=8)
    # test dataloader
    test_image = GraphemeDataset(test_data,
                                 reduced_test.consonant_diacritic.values,
                                 'test')
    batch_size = args.n_classes * args.n_samples
    test_loader = torch.utils.data.DataLoader(test_image,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=8)

    model = APINet(7).cuda()
    model = nn.DataParallel(model)

    try:
        model.load_state_dict(torch.load('DenseNet121root.pth'))
        print('loaded root pretrained weight')
    except:
        print('pretrained weight not found')

    epochs = 50
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4,
                                  weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0.
    for epoch in range(epochs):
        print('epochs {}/{} '.format(epoch+1, epochs))
        train_one_epoch(model, train_loader, optimizer)
        acc = validate(model, test_loader)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'DenseNet121root_NEW.pth')


if __name__ == '__main__':
    args = _init_()
    main(args)
