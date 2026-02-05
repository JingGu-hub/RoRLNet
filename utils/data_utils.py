from collections import Counter

import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

from utils.trend_season_decom import ts_decom
from utils.utils import build_dataset_pt, build_dataset_uea, flip_label


class TimeDataset(data.Dataset):
    def __init__(self, dataset, target, true_target):
        self.dataset = dataset
        self.target = target
        self.true_target = true_target

    def __getitem__(self, index):
        if self.true_target is not None:
            return self.dataset[index], self.target[index], self.true_target[index], index
        return self.dataset[index], self.target[index], index

    def __len__(self):
        return len(self.target)

def get_dataset(args):
    if args.archive == 'other':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_pt(args)
    elif args.archive == 'UEA':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_uea(args)
    input_dimension, seq_length = train_dataset.shape[1], train_dataset.shape[2]

    ts_aug = ts_decom(kernel_size=5, block_size=2, alpha=args.alpha, beta=args.beta)
    train_dataset, train_target = ts_aug(args, torch.from_numpy(train_dataset).type(torch.FloatTensor).cuda(), torch.from_numpy(train_target).type(torch.LongTensor).cuda())

    # corrupt label
    if args.label_noise_rate > 0:
        noise_train_target, mask_train_target = flip_label(args=args, dataset=train_dataset, target=train_target, ratio=args.label_noise_rate)

    # load train_loader
    train_loader = load_loader(args, train_dataset, noise_train_target, true_target=train_target)
    # load test_loader
    test_loader = load_loader(args, test_dataset, test_target, shuffle=False)

    return train_loader, test_loader, input_dimension, seq_length, num_classes

def load_loader(args, data, target, true_target=None, shuffle=True):
    dataset = TimeDataset(torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(target).type(torch.LongTensor), true_target=true_target)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)

    return loader