import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np


mat1 = np.arange(137)
mat2 = np.arange(236)


def Allign(image, partition='test'):
    ''' Align words to the center of image '''
    img1 = image.mean(axis=1)
    img2 = image.mean(axis=0)
    imgall = img1.mean()

    mean_x = int((img1 * mat1).mean() / imgall)
    mean_y = int((img2 * mat2).mean() / imgall)

    # shift it if it's training
    if partition == 'train':
        mean_x += np.random.randint(-16, 16)
        mean_y += np.random.randint(-16, 16)
        mean_x = np.clip(mean_x, 0, 137)
        mean_y = np.clip(mean_y, 0, 236)

    _t = min(68, mean_x)
    _b = min(69, 137 - mean_x)
    _l = min(118, mean_y)
    _r = min(118, 236 - mean_y)
    # print(_t, _b, _l, _r)

    zeros = np.zeros((137, 236))
    zeros[68-_t:68 + _b, 118-_l:118 + _r] = \
        image[mean_x - _t:mean_x + _b, mean_y-_l:mean_y + _r]

    return zeros[68 - 64: 68 + 64, 118 - 96: 118 + 96]


def AddNoise(image, scalar=(32./255)):
    ''' Add noise to image '''
    try:
        H, W = image.shape
        h = np.random.randint(1, H//2)
        w = np.random.randint(1, W//2)
        t = np.random.randint(0, H - h + 1)
        l = np.random.randint(0, W - w + 1)
        image[t:t+h, l:l+w] += (scalar * (np.random.rand(h, w) - 0.5))
    except:
        print('Add Noise ERROR')
    return image


def CutOff(image):
    ''' cut part of image and add noise block '''
    try:
        H, W = image.shape
        h = np.random.randint(1, H//4)
        w = np.random.randint(1, W//4)
        t = np.random.randint(0, H - h + 1)
        l = np.random.randint(0, W - w + 1)
        image[t:t+h, l:l+w] = np.random.rand(h, w) * 2 - 1
    except:
        print('Add Noise ERROR')
    return image


class GraphemeDataset(Dataset):
    ''' Dataset of Bengali '''
    def __init__(self, images, label, partition):
        self.partition = partition
        self.root = label

        img = images.iloc[:, 1:].values
        self.images = 255 - img.reshape(-1, 137, 236)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.root[idx]
        image = self.images[idx] / 255
        image = Allign(image, self.partition) * 2 - 1

        # Data Augmentation
        if self.partition == 'train':
            if np.random.rand() < 0.5:
                image = AddNoise(image, 64./255)
            if np.random.rand() < 0.5:
                image = CutOff(image)
            if np.random.rand() < 0.5:
                scalar = np.random.rand() + 0.5
                image = image * scalar + (scalar - 1)
            if np.random.rand() < 0.5:
                image = np.clip(image, -1, 1)

        return image, label


class BalancedBatchSampler(BatchSampler):
    ''' Batch Sampler for Training

    - n_classes : n_classes categories of objects in a batch
    - n_samples : n_samples of object for each class in a batch
    # batch_size is n_classes * n_samples

    '''

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.root
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        ''' Randomly Generate Batches of Training Data '''
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set,
                                       self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                st = self.used_label_indices_count[class_]
                ed = st + self.n_samples
                indices.extend(self.label_to_indices[class_][st: ed])
                self.used_label_indices_count[class_] = ed
                if ed + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
