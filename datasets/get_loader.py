import torch
import random
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader
from torchvision.transforms.transforms import Resize

from .celeba import CelebADataset
import datasets.transforms as SLT
from datasets.wildceleba import WildCelebA
from datasets.wildaflw import WildAFLW
from datasets.cub_mine import CUB2
from datasets.pascalvoc import PascalVoc
from datasets.penn_action import PennAction, PennActionDouble
from datasets.deepfashion import DeepFashion

from utils.utils import FeiWu

MAPDICT = {
    'celeba': CelebADataset,
    'wildceleba': WildCelebA,
    'wildaflw': WildAFLW,
    'cub': CUB2,
    'voc': PascalVoc,
    'penaction': PennAction,
    'bipenn': PennActionDouble,
    'deepfashion': DeepFashion,
}

def get_dataset_by_config(args):
    if 'TRAIN_SET' in args.DATASET:
        # get transform
        if args.DATASET.TRAIN_SET.get('pipeline', None) is not None:
            base_module = eval(args.DATASET.TRAIN_SET.Tname)
            train_transform = T.Compose([getattr(base_module, item['name'])(**item['paras']) \
                for item in args.DATASET.TRAIN_SET.pipeline])
        else:
            train_transform = T.Compose([
                    SLT.SLResize((args.DATASET.TRAIN_SET.data_rescale_height, args.DATASET.TRAIN_SET.data_rescale_width)),
                    SLT.SLToTensor(),
                    SLT.SLNormalize(mean=args.DATASET.TRAIN_SET.normalize_mean, std=args.DATASET.TRAIN_SET.normalize_std)
                ])

        # get dataset class
        assert args.DATASET.TRAIN_SET.dataset.lower() in MAPDICT.keys(), \
            'dataset %s isnot implemented' % args.DATASET.TRAIN_SET.dataset
        
        train_set = MAPDICT.get(args.DATASET.TRAIN_SET.dataset.lower())(
            transform=train_transform,
            **args.DATASET.TRAIN_SET.paras,
            args = args
        )

        # select a subset
        train_subset_item = args.DATASET.TRAIN_SET.get('subset_num', None)
        if train_subset_item is not None:
            if isinstance(train_subset_item, int):
                train_subset_item = list(range(train_subset_item))
            train_set = Subset(train_set, train_subset_item)
    else:
        train_set = None

    if 'TEST_SET' in args.DATASET:
        if args.DATASET.TEST_SET.get('pipeline', None) is not None:
            base_module = eval(args.DATASET.TEST_SET.Tname)
            test_transform = T.Compose([getattr(base_module, item['name'])(**item['paras']) \
                for item in args.DATASET.TEST_SET.pipeline])
        else:
            test_transform = T.Compose([
                    SLT.SLResize((args.DATASET.TEST_SET.data_rescale_height, args.DATASET.TEST_SET.data_rescale_width)),
                    SLT.SLToTensor(),
                    SLT.SLNormalize(mean=args.DATASET.TEST_SET.normalize_mean, std=args.DATASET.TEST_SET.normalize_std)
                ])
        assert args.DATASET.TEST_SET.dataset.lower() in MAPDICT.keys(), \
            'dataset %s isnot implemented' % args.DATASET.test_set.dataset
        test_set = MAPDICT.get(args.DATASET.TEST_SET.dataset.lower())(
            transform=test_transform,
            **args.DATASET.TEST_SET.paras,
            args = args
        )
        test_subset_item = args.DATASET.TEST_SET.get('subset_num', None)
        if test_subset_item is not None:
            if isinstance(test_subset_item, int):
                test_subset_item = list(range(test_subset_item))
            test_set = Subset(test_set, test_subset_item)
    else:
        test_set = None

    return train_set, test_set


def get_dataloader(args):
    train_set, test_set = get_dataset_by_config(args)

    if args.get('distributed', False):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, 
            num_replicas=args.world_size,
            rank=args.rank
        )
        # test_sampler = torch.utils.data.distributed.DistributedSampler(
        #     test_set, 
        #     num_replicas=args.world_size,
        #     rank=args.rank
        # )
    else:
        train_sampler, test_sampler = None, None

    if train_set is None:
        train_loader, train_sampler = None, None
    else:
        # get train and test dataloader
        if 'shuffle' in args.DATASET.TRAIN_SET:
            shuffle = args.DATASET.TRAIN_SET.shuffle
        else:
            shuffle = train_sampler is None
        train_loader = DataLoader(
            dataset = train_set, 
            batch_size = args.DATASET.TRAIN_SET.batch_size, 
            drop_last = args.DATASET.TRAIN_SET.get('drop_last', False),
            shuffle = shuffle,
            num_workers = args.DATASET.TRAIN_SET.num_workers,
            pin_memory = args.DATASET.TRAIN_SET.get('pin_memory', True),
            sampler = train_sampler
        )
    
    if test_set is None:
        test_loader = None
    else:
        test_loader = DataLoader(
            dataset = test_set, 
            batch_size = args.DATASET.TEST_SET.batch_size, 
            drop_last = args.DATASET.TRAIN_SET.get('drop_last', False),
            shuffle = False,
            num_workers = args.DATASET.TEST_SET.num_workers,
            pin_memory = args.DATASET.TEST_SET.get('pin_memory', True),
        )
    
    return train_loader, test_loader, train_sampler