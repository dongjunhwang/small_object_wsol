"""
Original code : https://github.com/zh460045050/DA-WSOL_CVPR2022
"""

import numpy as np
import torch
import torch.nn as nn
import random

from sklearn.cluster import k_means
from torch.autograd import Variable


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if source.size()[0] != target.size()[0]:
        if source.size()[0] < target.size()[0]:
            source = source.unsqueeze(0)
            source = source.expand((np.int64(target.size()[0] / source.size()[1]), source.size()[1], source.size()[2]))
            source = source.contiguous().view(target.size())
        else:
            target = target.unsqueeze(0)
            target = target.expand((np.int64(source.size()[0] / target.size()[1]), target.size()[1], target.size()[2]))
            target = target.contiguous().view(source.size())

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def cal_mmd(batch_source, batch_target, count):
    loss_mmd = Variable(torch.zeros(1)).cuda()
    for i in range(0, count):
        loss_mmd += mmd_rbf_accelerate(batch_source[i, :, :], batch_target[i, :, :])
    loss_mmd /= count

    return loss_mmd


class DACAM(nn.Module):
    def __init__(self, **kwargs):
        super(DACAM, self).__init__()
        _BETA = {"ILSVRC": 0.3, "CUB": 0.3, "OpenImages": 0.2}
        _UNIVER = {"ILSVRC": 3, "CUB": 2, "OpenImages": 3}
        self.num_classes = kwargs['num_classes']
        self.architecture = kwargs['architecture']
        self.model = kwargs['model']
        dataset_name = kwargs['dataset_name']

        # For fuse the crop method.
        self.beta = _BETA[dataset_name]
        self.univer = _UNIVER[dataset_name]
        self.feature_dims = 1024 if self.architecture != 'resnet50' else 2048

        self.TSA = TSA(self.num_classes, self.feature_dims)
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    def forward(self, images, target, crop=False, return_cam=False):
        output_dict = self.model(images, target, da=True, return_cam=return_cam)
        if return_cam:
            return output_dict
        pixel_feature = output_dict['pixel_feature']
        image_feature = output_dict['image_feature']


        S_T_f, T_t, T_u, _, count, mask = self.TSA.samples_split(image_feature, pixel_feature, target)

        logits = output_dict['logits']
        loss_c = self.cross_entropy_loss(logits, target)
        loss_u = torch.mean(torch.mean(T_u))
        loss_d = cal_mmd(S_T_f, T_t, count)
        loss = loss_c + self.beta * loss_d + self.univer * loss_u
        if crop:
            loss = self.beta * loss_d + self.univer * loss_u
            return {'cam_weights': output_dict['cam_weights'],
                    'logits': logits, 'feature_map': output_dict['feature_map'],
                    'loss': loss}
        return logits, loss


class TSA(object):
    def __init__(self, num_classes, feature_dims, sample_num_source=32, sample_num_target=32):
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.cluster_centers = np.zeros((self.num_classes, self.feature_dims)) #M_{:,1:}
        self.cluster_counts = np.zeros((self.num_classes, 1)) #r_{1:}
        self.universe_centers = np.zeros((1, self.feature_dims)) #M_{:,0}
        self.universe_count = 0 #r_{0}
        self.sample_num_source = sample_num_source
        self.sample_num_target = sample_num_target

    def samples_split(self, image_feature, pixel_feature, target):

        source = image_feature.view(-1, image_feature.shape[1])
        mask = np.zeros( (source.shape[0], pixel_feature.shape[2], pixel_feature.shape[3]) )
        batch_source = torch.zeros(source.shape[0], self.sample_num_source, image_feature.shape[1]).cuda()
        batch_target = torch.zeros(source.shape[0], self.sample_num_target, image_feature.shape[1]).cuda()
        batch_label = torch.zeros(source.shape[0]).cuda()
        count = 0
        count_u = 0
        for i in range(0, source.shape[0]):
            batch_source[count, 0, :] = source[i, :]
            lab = target[i]
            all_samples = pixel_feature.permute(0, 2, 3, 1).contiguous()[i, :, :, :].view(-1, source.shape[1])
            samples = all_samples.detach().clone().cpu().numpy()

            if self.cluster_counts[lab] == 0:
                self.cluster_counts[lab] = self.cluster_counts[lab] + 1
                self.cluster_centers[lab, :] = source[i, :].unsqueeze(0).detach().clone().cpu().numpy() / self.cluster_counts[lab] + self.cluster_centers[lab, :] * ((self.cluster_counts[lab]-1) / self.cluster_counts[lab] )

            #### Kmeans Cluster
            center_inits = torch.cat( [torch.from_numpy(self.universe_centers).cuda(), source[i, :].unsqueeze(0), torch.from_numpy(self.cluster_centers)[lab, :].unsqueeze(0).cuda()], dim=0).detach().clone().cpu().numpy()
            center, label, pb = k_means(samples, n_clusters=3, init=center_inits, n_init=1, random_state=0)

            #### Update Cache Matrix
            self.cluster_counts[lab] = self.cluster_counts[lab] + 1
            self.cluster_centers[lab, :] = np.expand_dims(center[1, :], axis=0) / self.cluster_counts[lab] + self.cluster_centers[lab, :] * ((self.cluster_counts[lab] - 1) / self.cluster_counts[lab] )

            self.universe_count = self.universe_count + 1
            self.universe_centers[0, :] = np.expand_dims(center[0, :], axis=0) / self.universe_count + self.universe_centers[0, :] * ((self.universe_count - 1) / self.universe_count )

            #### Sample Source/Target/Univers Items
            cur_univer = all_samples[label == 0, :]
            cur_source = all_samples[label == 1, :]
            cur_target = all_samples[label == 2, :]

            #### Random Sampling
            rand_index_target = np.arange(0, cur_target.shape[0])
            random.shuffle(rand_index_target)
            rand_index_source = np.arange(0, cur_source.shape[0])
            random.shuffle(rand_index_source)

            if len(rand_index_target) >= self.sample_num_target and len(rand_index_source) >= (self.sample_num_source - 1):
                batch_source[count, 1:, :] = torch.index_select(cur_source, 0, torch.from_numpy(rand_index_source)[:min(self.sample_num_source-1, len(rand_index_source))].cuda())
                batch_target[count, :, :] = torch.index_select(cur_target, 0, torch.from_numpy(rand_index_target)[:min(self.sample_num_target, len(rand_index_target))].cuda())
                batch_label[count] = target[i]
                count = count + 1
            if count_u == 0:
                batch_universum = cur_univer.view(-1, image_feature.shape[1])
                count_u = batch_universum.shape[1]
            else:
                batch_universum = torch.cat([batch_universum, cur_univer.view(-1, image_feature.shape[1])], dim=0)

            label = np.float32(label)
            label[label == 2] = 0.5
            mask[i, :, :] = label.reshape((pixel_feature.shape[2], pixel_feature.shape[3]))

        return batch_source, batch_target, batch_universum, batch_label, count, mask