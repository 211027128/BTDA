import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
import numpy as np
import json
import os
import math
import random
from datasets.DistributedProxySampler import DistributedProxySampler

def split_imb_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    data, target = np.array(data), np.array(target)

    label, nums = np.unique(target, return_counts=True)  #
    nums_sort_index = np.argsort(nums, kind='stable')
    # print(nums_sort_index)

    imb_factor = args.imb_factor
    img_max = nums[nums_sort_index[0]]
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (0.02 ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(max(int(num), 1))
    lb_num = img_num_per_cls/ np.sum(img_num_per_cls) * num_labels
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = (img_max-lb_num[0]) * (imb_factor ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(max(int(num), 1))

    
    p_target=[]
    q_target=[]
    '''
    #for true distribution
    a=[]
    random.seed(0)
    noisy=np.random.rand(10)
    '''
    for i in range(10):
      p_target.append(img_num_per_cls[i]/np.sum(img_num_per_cls))
      q_target.append(lb_num[i]/np.sum(lb_num))
    '''
    #for true distribution
    for i in range(10):
      a.append(p_target[i]+noisy[i]*0)
    #print(noisy,a)
    p_target=[]
    for i in range(10):
      p_target.append(a[i]/np.sum(a))    
    print('p_target:',p_target)
    print('q_target:',q_target)
    '''

    lb_data = []
    lbs = []
    ulb_data=[]
    ulbs=[]
    # lb_idx = []
    for i in range(num_classes):
        idx = np.where(target == i)[0]
        # idx = np.random.choice(idx, img_num_per_cls[c], False)
        idx_l = idx[:math.ceil(lb_num[i])]
        # lb_idx.extend(idx)
        lb_data.extend(data[idx_l])
        lbs.extend(target[idx_l])
        idx_u= idx[math.ceil(lb_num[i]):math.ceil(lb_num[i])+img_num_per_cls[i]]
        ulb_data.extend(data[idx_u])
        ulbs.extend(target[idx_u])
    lb_data, lbs ,ulb_data, ulbs,= np.array(lb_data), np.array(lbs),np.array(ulb_data), np.array(ulbs)
    #
    #lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, num_classes, index,im_num=img_num_per_cls)

    #ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    print( len(lb_data), lb_num[1]/lb_num[2])
    print( len(ulb_data), img_num_per_cls[1]/img_num_per_cls[2])
    return lb_data, lbs, ulb_data,ulbs, p_target,q_target

def split_ssl_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
                                                
    label, nums = np.unique(target, return_counts=True)  #
    nums_sort_index = np.argsort(nums,kind='stable')
    #print(nums_sort_index)

    #
    imb_factor = args.imb_factor
    img_max = nums[nums_sort_index[0]]
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(max(int(num), 1))

    temp = np.zeros_like(nums_sort_index)
    for i in range(len(nums_sort_index)):
        temp[nums_sort_index[i]] = img_num_per_cls[i]

    img_num_per_cls = temp
    #print("img_num_per_cls",img_num_per_cls)

    lb_data = []
    lbs = []
    # lb_idx = []
    for i in range(num_classes):
        idx = np.where(target == i)[0]
        # idx = np.random.choice(idx, img_num_per_cls[c], False)
        idx = idx[:img_num_per_cls[i]]
        # lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])
    data, target = np.array(lb_data), np.array(lbs)
    #
    lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, num_classes, index,
                                                im_num=img_num_per_cls)
    
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    print(len(data),len(lb_data),len(ulb_idx))
    #include_lb_to_ulb=False
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]

def sample_labeled_data(args, data, target,
                        num_labels, num_classes,
                        index=None, name=None, im_num=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    dump_path = os.path.join(args.save_dir, args.save_name, 'sampled_label_idx.npy')

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_data = data[lb_idx]
        lbs = target[lb_idx]
        return lb_data, lbs, lb_idx

    #
    # ratio = num_labels / num_classes / max(im_num)
    # samples_per_class = np.array(im_num) * ratio

    samples_per_class = im_num / np.sum(im_num) * num_labels
    print(np.sum(im_num),samples_per_class)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        # idx = np.random.choice(idx, math.ceil(samples_per_class[c]), False)
        idx = np.random.choice(idx, math.ceil(samples_per_class[c]), False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])
    #

    # samples_per_class = int(num_labels / num_classes)
    #
    # lb_data = []
    # lbs = []
    # lb_idx = []
    # for c in range(num_classes):
    #     idx = np.where(target == c)[0]
    #     idx = np.random.choice(idx, samples_per_class, False)
    #     lb_idx.extend(idx)
    #
    #     lb_data.extend(data[idx])
    #     lbs.extend(target[idx])

    np.save(dump_path, np.array(lb_idx))

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__
                               if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_data_loader(dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler=None,
                    replacement=True,
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == 'RandomSampler':
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
