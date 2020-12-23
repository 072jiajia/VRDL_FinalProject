import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np
import pandas as pd
from torchvision import models
from skimage.transform import AffineTransform, warp
import cv2


def pdist(vectors):
    ''' Compute squares of Pair-Wise Distances
    Input:
        - vectors : (batch_size, num_feature)
    Output:
        - distance_matrix : (batch_size, batch_size)
    As we can compute the distance in euclidean space by
        (vx - vy) ** 2 = vx * vx + vy * vy - 2 * vx * vy
    Compute  vx * vx  and  vy * vy  by element-wise multiplication
            vectors.pow(2).sum(dim=1)
    And compute  2 * vx * vy  by matrix multiplication
            2 * vectors.mm(torch.t(vectors))
    '''
    v2 = vectors.pow(2).sum(dim=1)
    xy2 = 2 * vectors.mm(torch.t(vectors))
    distance_matrix = v2.view(1, -1) - xy2 + v2.view(-1, 1)
    return distance_matrix


class APINet(nn.Module):
    ''' Module of API_Net
    *** Using densenet121 as its backbone ***
    *** Parameters ***
    - conv       : CNN backbone of this module
    - map        : mlp to compute contrastive clues
    - fc         : fully-connected layer (last layer)
    '''

    def __init__(self, out_channel):
        super(APINet, self).__init__()

        backbone = models.densenet121(pretrained=True)
        backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7),
                                            stride=(2, 2), padding=(3, 3))
        num_feature = backbone.classifier.in_features
        layers = list(backbone.children())[:-1]
        self.conv = nn.Sequential(*layers)

        embedding = 512
        self.map = nn.Sequential(
            nn.Linear(num_feature * 2, embedding),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embedding, embedding),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embedding, num_feature)
        )

        self.fc = nn.Linear(num_feature, out_channel)

        self.sigmoid = nn.Sigmoid()
        self.softmax_layer = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss()
        self.rank_criterion = nn.MarginRankingLoss(margin=0)

    def get_pairs(self, embeddings, labels):
        ''' Compute the Intra Pairs and the Inter Pairs
        * intra : The most similar object which is in the same catagory
        * inter : The most similar object which is not in the same catagory
        Use euclidean distance in the feature space
            to determine the similarity between objects
        '''
        distance_matrix = pdist(embeddings).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().reshape(-1, 1)

        # lb_eqs is a batch_size * batch_size boolean matrix about
        # whether object_n and object_m is in the same category
        lb_eqs = (labels == labels.T)

        # If we find the most similar object directly, We might
        # just find out that itself it's the most similar object
        # So use np.diag_indices to easily handle this problem
        dia_inds = np.diag_indices(labels.shape[0])

        # Compute the intra / inter pairs
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs is False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs is True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        # Generate the return data
        intra_pairs = torch.from_numpy(intra_idxs).long().cuda()
        inter_pairs = torch.from_numpy(inter_idxs).long().cuda()

        return intra_pairs, inter_pairs

    def fine_grained_learning(self, features, labels):
        intra_pairs, inter_pairs = self.get_pairs(features, labels)
        features1 = torch.cat([features, features], dim=0)
        features2 = torch.cat([features[intra_pairs],
                               features[inter_pairs]], dim=0)
        mutual_features = torch.cat([features1, features2], dim=1)
        map_out = self.map(mutual_features)

        gate1 = torch.mul(map_out, features1)
        gate1 = self.sigmoid(gate1)
        gate2 = torch.mul(map_out, features2)
        gate2 = self.sigmoid(gate2)

        # Obtain attentive features via residual attention
        # * self  : feature vector activated by its own gate
        # * other : feature vector activated by the gate of paired image
        features1_self = torch.mul(gate1, features1) + features1
        features1_other = torch.mul(gate2, features1) + features1
        features2_self = torch.mul(gate2, features2) + features2
        features2_other = torch.mul(gate1, features2) + features2

        class1_self = self.fc(features1_self)
        class1_other = self.fc(features1_other)
        class2_self = self.fc(features2_self)
        class2_other = self.fc(features2_other)

        # labels of objects in pairs
        # return it to compute MarginRankingLoss
        labels1 = torch.cat([labels, labels], dim=0)
        labels2 = torch.cat([labels[intra_pairs],
                             labels[inter_pairs]], dim=0)

        # Cross Entropy loss
        classes = torch.cat([class1_self, class1_other,
                             class2_self, class2_other], dim=0)
        concat_labels = torch.cat([labels1, labels1,
                                   labels2, labels2], dim=0)

        CE = self.CE(classes, concat_labels)

        # MarginRankingLoss, the score determined by intra pairs
        # should be higher then inter pairs
        self_logits = torch.cat([class1_self, class2_self], dim=0)
        other_logits = torch.cat([class1_other, class2_other], dim=0)

        obj_idx = torch.arange(4 * features.shape[0]).cuda().long()
        label_idx = torch.cat([labels1, labels2], dim=0)

        self_scores = self.softmax_layer(self_logits)[obj_idx, label_idx]
        other_scores = self.softmax_layer(other_logits)[obj_idx, label_idx]

        flag = torch.ones([4 * features.shape[0], ]).cuda()
        rank_loss = self.rank_criterion(self_scores, other_scores, flag)

        # compute total loss
        return CE + rank_loss

    def forward(self, images, label=None):
        ''' Forward Propageting Function
        - images : Input image
        - label  : When it is None, make label's prediction
                   else, do discriminative feature learning
        '''
        conv_out = self.conv(images)
        pool_out = F.adaptive_avg_pool2d(conv_out, (1, 1)).squeeze()

        if label is None:
            return self.fc(pool_out)
        else:
            return self.fine_grained_learning(pool_out, label)
