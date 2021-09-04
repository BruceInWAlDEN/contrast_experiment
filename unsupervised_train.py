'''
本脚本用于无监督训练， 获得预训练权重

'''
from model import models
from gaussian_blur import GaussianBlur

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision.transforms import transforms

import glob
import os
import argparse
import datetime
import numpy as np
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--weight', default="", type=str)
parser.add_argument('--temperature', default=0.07, type=float)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, s, e):
        self.dir_path = data_path
        self.data = []      # [imgpath1, imgpath2...] str
        self.label = []     # [imglabel1, imglabel2...] int
        self.transform = get_simclr_pipeline_transform(size=256)

        paths = glob.glob(os.path.join(data_path, '*.txt'))
        for path in paths:
            with open(path, 'r') as file:
                lines = file.readlines()
                num, total, label = lines[0].split(',')
                data = lines[1 + int(num)*s: 1 + int(num)*e]
                data = [x.strip('\n') for x in data if x.strip('\n') is not None]
            self.data += data
            self.label += [int(label)]*len(data)

        print('DATA SIZE == {}'.format(len(self.data)))

    def __getitem__(self, item):
        img = get_img(self.data[item])

        return [self.transform(img) for i in range(2)]

    def __len__(self):
        return len(self.data)


def main():
    args = parser.parse_args()

    # model
    model = models.ResNetSimCLR(128)
    model = nn.DataParallel(model)
    model = model.cuda()

    # weight
    if args.weight:
        pth = torch.load(args.weight)
        model.load_state_dict(pth)
        print('weight {} loaded !'.format(args.weight))

    # data
    train_data = MyDataset('data', s=0, e=16)  #  0-4 4-8 8-12 12-16       16-20  20-22
    test_data = MyDataset('data', s=16, e=20)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), 0.0003, weight_decay=1e-4)

    # metric
    metric = {}
    metric['epoch'] = []
    metric['train_batch'] = []
    metric['test_batch'] = []
    metric['train_batch_loss'] = []
    metric['train_epoch_loss'] = []
    metric['test_batch_loss'] = []
    metric['test_epoch_loss'] = []

    metric['train_batch_top1'] = []
    metric['train_batch_top5'] = []
    metric['test_batch_top1'] = []
    metric['test_batch_top5'] = []

    for epoch in range(args.epochs):
        metric['epoch'] = [epoch]
        # train
        print("==== TRAINING ==========================================")
        start_time = datetime.datetime.now()
        model.train()
        epoch_loss = 0
        for i, images in enumerate(train_loader):
            print("TRAINING BATCH:::  {}".format(i+1))
            images = torch.cat(images, dim=0)
            images = images.cuda()   # 2*B, C=3, 256, 256
            features = model(images)
            logits, labels = info_nce_loss(features, args)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            if i % 10 == 0:
                # write metric
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                metric['train_batch'].append(i)
                metric['train_batch_loss'].append(loss)
                metric['train_batch_top1'].append(top1[0])
                metric['train_batch_top5'].append(top5[0])
                print('BATCH TRAIN LOSS::: {}, top1::{}, top5::{}'.format(loss, top1[0], top5[0]))

        metric['train_epoch_loss'].append(epoch_loss/i)
        end_time = datetime.datetime.now()
        print("TRAINING EPOCH:::  {}/{} LOSS::: {}  TIME COST::: {} s".format(epoch + 1, args.epochs, epoch_loss/i, (end_time-start_time).seconds))

        # save metric
        torch.save(metric, 'metric')

        # save model
        pth = model.state_dict()
        torch.save(pth, 'check/check_{}.pth'.format(epoch+1))

        # test
        print('==== TESTING ==========================================')
        start_time = datetime.datetime.now()
        model.eval()
        epoch_loss = 0
        for i, images in enumerate(test_loader):
            print("TESTING BATCH:::  {}".format(i+1))
            images = torch.cat(images, dim=0)
            images = images.cuda()   # 2*B, C=3, 256, 256
            # -------------------------------------------
            with torch.set_grad_enabled(False):
                features = model(images)     # feature 128
                logits, labels = info_nce_loss(features, args)
                loss = criterion(logits, labels)
            # -------------------------------------------
            epoch_loss += loss
            if i % 10 == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                metric['test_batch'].append(i)
                metric['test_batch_loss'].append(loss)
                metric['test_batch_top1'].append(top1[0])
                metric['test_batch_top5'].append(top5[0])
                print('BATCH TEST LOSS::: {}, top1::{}, top5::{}'.format(loss, top1[0], top5[0]))

        print('EPOCH TEST LOSS::: {}'.format(epoch_loss/i))
        metric['test_epoch_loss'].append(epoch_loss/i)
        end_time = datetime.datetime.now()
        print("TEST EPOCH:::  {}/{} LOSS::: {}  TIME COST::: {} s".format(epoch + 1, args.epochs, epoch_loss/i, (end_time-start_time).seconds))

        # save model
        torch.save(metric, 'check/unsupervised_metric')


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor()])
    return data_transforms


def info_nce_loss(features, args):
        labels = torch.cat([torch.arange(args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits = logits / args.temperature

        return logits, labels


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_img(path):
    '''
    :param path: 图片路径
    :return: resize 中心裁剪 numpy
    '''
    img = Image.open(path)
    img = img.resize((768, 768), Image.ANTIALIAS)
    box = (128, 128, 128+512, 128+512)
    img = img.crop(box)

    return img.convert('RGB')


if __name__ == '__main__':
    main()
