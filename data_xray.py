"""Data provider"""

import torch
import torch.utils.data as data

import os
import nltk
import numpy as np

from termcolor import colored

FRONTAL_POS_ENC = 0.05
LATERAL_POS_ENC = -0.05

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, view):

        loc = data_path + '/'
        print(colored("Fetching the {} images".format(view), 'magenta'))

        # load the captions features
        if data_split == 'train':
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # resnet
            self.images = np.load(loc + '%s_ims_vgg_%s.npy' % (data_split, view), mmap_mode='r')

        elif data_split == 'dev':
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # load the image features
            self.images = np.load(loc + 'valid_ims_vgg_%s.npy' % view, mmap_mode='r')

        else:
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # load the image features
            self.images = np.load(loc + '%s_ims_vgg_%s.npy' % (data_split, view), mmap_mode='r')

        self.length = len(self.captions)
        # load the captions lengths
        self.lengths = np.load(loc + '%s_len_frontal.npy' % data_split, mmap_mode='r')
        self.im_div = 1

        for i, val in enumerate(self.lengths):
            if (val == 0):
                print("deleting sample number {}".format(i))
                self.captions = np.delete(self.captions,[i],0)
                self.lengths = np.delete(self.lengths,[i],0)
                self.images = np.delete(self.images,[i],0)


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])

        target = torch.Tensor(self.captions[index])
        length = self.lengths[index]
        return image, target, index, img_id, length

    def __len__(self):
        return self.length

class PrecompDatasetDouble(data.Dataset):
    """
    Load precomputed captions and image features for double model (frontal and lateral)
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split):

        loc = data_path + '/'
        print(colored("Fetching Frontal and Lateral images", 'magenta'))
        # load the captions features
        if data_split == 'train':
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # resnet
            self.imagesF = np.load(loc + '%s_ims_vgg_frontal.npy' % data_split, mmap_mode='r')
            self.imagesL = np.load(loc + '%s_ims_vgg_lateral.npy' % data_split, mmap_mode='r')
        elif data_split == 'dev':
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # load the image features
            self.imagesF = np.load(loc + 'valid_ims_vgg_frontal.npy', mmap_mode='r')
            self.imagesL = np.load(loc + 'valid_ims_vgg_lateral.npy', mmap_mode='r')

        else:
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # load the image features
            self.imagesF = np.load(loc + '%s_ims_vgg_frontal.npy' % data_split, mmap_mode='r')
            self.imagesL = np.load(loc + '%s_ims_vgg_lateral.npy' % data_split, mmap_mode='r')

        self.length = len(self.captions)
        # load the captions lengths
        self.lengths = np.load(loc + '%s_len_frontal.npy' % data_split, mmap_mode='r')
        self.im_div = 1

        for i, val in enumerate(self.lengths):
            if (val == 0):
                print("deleting sample number {}".format(i))
                self.captions = np.delete(self.captions,[i],0)
                self.lengths = np.delete(self.lengths,[i],0)
                self.imagesF = np.delete(self.imagesF,[i],0)
                self.imagesL = np.delete(self.imagesL, [i], 0)


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        imageF = torch.Tensor(self.imagesF[img_id])
        imageL = torch.Tensor(self.imagesL[img_id])
        target = torch.Tensor(self.captions[index])
        length = self.lengths[index]
        return imageF, imageL, target, index, img_id, length

    def __len__(self):
        return self.length


class PrecompDatasetCat(data.Dataset):
    """
    Load precomputed captions and image features for double model (frontal and lateral)
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, model_type, img_dim):

        loc = data_path + '/'
        print(colored("Fetching Frontal and Lateral images", 'magenta'))
        # load the captions features
        if data_split == 'train':
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # resnet
            frontal_images = np.load(loc + '%s_ims_vgg_frontal.npy' % data_split, mmap_mode='r+')
            lateral_images = np.load(loc + '%s_ims_vgg_lateral.npy' % data_split, mmap_mode='r+')
           #self.images = np.concatenate([np.load(loc + '%s_ims_vgg_frontal.npy' % data_split, mmap_mode='r'),
           #                          np.load(loc + '%s_ims_vgg_lateral.npy' % data_split, mmap_mode='r')], axis=1)

        elif data_split == 'dev':
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # load the image features
            frontal_images = np.load(loc + 'valid_ims_vgg_frontal.npy', mmap_mode='r+')
            lateral_images = np.load(loc + 'valid_ims_vgg_lateral.npy', mmap_mode='r+')
            #self.images = np.concatenate([np.load(loc + 'valid_ims_vgg_frontal.npy', mmap_mode='r'),
            #                         np.load(loc + 'valid_ims_vgg_lateral.npy', mmap_mode='r')], axis=1)
        else:
            self.captions = np.load(loc + '%s_caps_frontal.npy' % data_split, mmap_mode='r')
            # load the image features
            frontal_images = np.load(loc + '%s_ims_vgg_frontal.npy' % data_split, mmap_mode='r+')
            lateral_images = np.load(loc + '%s_ims_vgg_lateral.npy' % data_split, mmap_mode='r+')
            #self.images = np.concatenate([np.load(loc + '%s_ims_vgg_frontal.npy' % data_split, mmap_mode='r'),
            #                         np.load(loc + '%s_ims_vgg_lateral.npy' % data_split, mmap_mode='r')], axis=1)

        self.length = len(self.captions)
        # load the captions lengths
        self.lengths = np.load(loc + '%s_len_frontal.npy' % data_split, mmap_mode='r')
        self.im_div = 1

        for i, val in enumerate(self.lengths):
            if (val == 0):
                print("deleting sample number {}".format(i))
                self.captions = np.delete(self.captions,[i],0)
                self.lengths = np.delete(self.lengths,[i],0)
                #self.images = np.delete(self.images,[i],0)
                frontal_images = np.delete(frontal_images,[i],0)
                lateral_images = np.delete(lateral_images,[i],0)

        if model_type == 'pos_enc_const_model':
            frontal_images += FRONTAL_POS_ENC
            lateral_images += LATERAL_POS_ENC

        if model_type == 'pos_enc_vec_model': #add values in [-0.098, 0.098] not including zero.
            for i in range(49):
                frontal_images[:,i,:] += (i+1)/500
                lateral_images[:,i,:] += -(i+1)/500


        self.images = np.concatenate([frontal_images, lateral_images], axis=1)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        target = torch.Tensor(self.captions[index])
        length = self.lengths[index]
        return image, target, index, img_id, length

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, lengths = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)
    targets = torch.stack(captions, 0)

    return images, targets, lengths, ids

def collate_fn_double(data):
    """
    Build mini-batch tensors from a list of (imageF, imageL, caption, index, img_id) tuples.
    Args:
        data: list of (imageF, imageL, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - imagesF: torch tensor of shape (batch_size, 36, 2048).
        - imagesL: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    imagesF, imagesL, captions, ids, img_ids, lengths = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    imagesF = torch.stack(imagesF, 0)
    imagesL = torch.stack(imagesL, 0)
    targets = torch.stack(captions, 0)

    return imagesF, imagesL, targets, lengths, ids

def get_precomp_loader(data_path, data_split, opt, batch_size=100,
                       shuffle=True, num_workers=2, view="frontal", model_type='regular_model'):
    print(colored("loading data for {}, with model type {}".format(data_split, model_type), 'green'))

    if model_type == 'regular_model':
        dset = PrecompDataset(data_path, data_split, view)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)

    elif model_type in ['double_model', 'light_double_model', 'pretrained_double_model', 'frozen_double_model']:
        dset = PrecompDatasetDouble(data_path, data_split)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn_double)

    elif model_type in ['cat_model', 'tag_model', 'pos_enc_const_model', 'pos_enc_vec_model', 'pos_enc_sin_model']:
        dset = PrecompDatasetCat(data_path, data_split, model_type, opt.img_dim)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
    else:
        raise Exception("Unsupported model type")

    return data_loader


def get_loaders(data_name, batch_size, workers, model_type, opt):
    # get the data path
    dpath = opt.data_path

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', opt,
                                      batch_size, True, workers, opt.view, model_type)


    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', opt,
                                    100, False, workers, opt.view, model_type)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size, workers, opt, view='frontal', model_type="regular_model"):
    # get the data path
    dpath = opt.data_path
    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, opt,
                                     100, False, workers, view, model_type)
    return test_loader
