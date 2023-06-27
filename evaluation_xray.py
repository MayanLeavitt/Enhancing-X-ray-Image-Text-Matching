"""Evaluation"""

from __future__ import print_function
import os
import sys
import time

import torch
import numpy as np
import pickle as pl

from data_xray import get_test_loader
from model_xray import SGRAF, DoubleSGRAF, lightDoubleSGRAF
from collections import OrderedDict
import opts_xray

from termcolor import colored

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    #image, target, index, img_id, length
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)

        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), 97, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        # cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions
    return img_embs, cap_embs, cap_lens


def encode_data_double(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embsF = None
    cap_embsF = None
    img_embsL = None
    cap_embsL = None

    max_n_word = 0
    # image, target, index, img_id, length
    for i, (imagesF, imagesL, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (imagesF, imagesL, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_embF, cap_embF, img_embL, cap_embL, cap_len = model.forward_emb(imagesF, imagesL, captions, lengths)

        if (img_embsF is None) or (img_embsL is None):
            img_embsF = np.zeros((len(data_loader.dataset), img_embF.size(1), img_embF.size(2)))
            cap_embsF = np.zeros((len(data_loader.dataset), 97, cap_embF.size(2)))
            img_embsL = np.zeros((len(data_loader.dataset), img_embL.size(1), img_embL.size(2)))
            cap_embsL = np.zeros((len(data_loader.dataset), 97, cap_embL.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embsF[ids] = img_embF.data.cpu().numpy().copy()
        cap_embsF[ids] = cap_embF.data.cpu().numpy().copy()
        img_embsL[ids] = img_embL.data.cpu().numpy().copy()
        cap_embsL[ids] = cap_embL.data.cpu().numpy().copy()
        # cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del imagesF, imagesL, captions
    return img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens


def encode_data_light_double(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embsF = None
    img_embsL = None
    cap_embs = None

    max_n_word = 0
    # image, target, index, img_id, length
    for i, (imagesF, imagesL, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (imagesF, imagesL, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_embF, img_embL, cap_emb, cap_len = model.forward_emb(imagesF, imagesL, captions, lengths)
        if (img_embsF is None) or (img_embsL is None):
            img_embsF = np.zeros((len(data_loader.dataset), img_embF.size(1), img_embF.size(2)))
            img_embsL = np.zeros((len(data_loader.dataset), img_embL.size(1), img_embL.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), 97, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embsF[ids] = img_embF.data.cpu().numpy().copy()
        img_embsL[ids] = img_embL.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        # cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del imagesF, imagesL, captions
    return img_embsF, img_embsL, cap_embs, cap_lens


def evalrank(model_path, view, data_path=None, split='test', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    save_epoch = checkpoint['epoch']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path


    # construct model
    model = SGRAF(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt, view, opt.model_type)
    print(colored("=> loaded checkpoint_epoch {}".format(save_epoch), 'red'))
    #
    print('Computing results...')
    img_embs, cap_embs, cap_lens = encode_data(model, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0], cap_embs.shape[0]))

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end - start)

    # bi-directional retrieval
    r, rt = i2t(sims, return_ranks=True)
    ri, rti = t2i(sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    return rti,rt

def evalrank_double(model_path, data_path=None, split='test', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    save_epoch = checkpoint['epoch']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path


    # construct model
    model = DoubleSGRAF(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt, model_type=opt.model_type)
    print(colored("=> loaded checkpoint_epoch {}".format(save_epoch), 'red'))
    #
    print('Computing results...')
    img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens = encode_data_double(model, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embsF.shape[0], cap_embsF.shape[0]))

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores_double(model, img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end - start)

    # bi-directional retrieval
    r, rt = i2t(sims, return_ranks=True)
    ri, rti = t2i(sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    return rti,rt

def evalrank_light_double(model_path, data_path=None, split='test', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    save_epoch = checkpoint['epoch']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path


    # construct model
    model = lightDoubleSGRAF(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt, model_type=opt.model_type)
    print(colored("=> loaded checkpoint_epoch {}".format(save_epoch), 'red'))
    #
    print('Computing results...')
    img_embsF, img_embsL, cap_embs, cap_lens = encode_data_light_double(model, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embsF.shape[0], cap_embs.shape[0]))

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores_light_double(model, img_embsF, img_embsL, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end - start)

    # bi-directional retrieval
    r, rt = i2t(sims, return_ranks=True)
    ri, rti = t2i(sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    return rti,rt

def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def shard_attn_scores_double(model, img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens, opt, shard_size=100):
    n_im_shard = (len(img_embsF) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embsF) - 1) // shard_size + 1
    sims = np.zeros((len(img_embsF), len(cap_embsF)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embsF))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embsF))

            with torch.no_grad():
                imF = torch.from_numpy(img_embsF[im_start:im_end]).float().cuda()
                caF = torch.from_numpy(cap_embsF[ca_start:ca_end]).float().cuda()
                imL = torch.from_numpy(img_embsL[im_start:im_end]).float().cuda()
                caL = torch.from_numpy(cap_embsL[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(imF, caF, imL, caL, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def shard_attn_scores_light_double(model, img_embsF, img_embsL, cap_embs, cap_lens, opt, shard_size=100):
    n_im_shard = (len(img_embsF) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
    sims = np.zeros((len(img_embsF), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embsF))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                imF = torch.from_numpy(img_embsF[im_start:im_end]).float().cuda()
                imL = torch.from_numpy(img_embsL[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(imF, imL, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims


def i2t(sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5))
    top10 = np.zeros((npts, 10))

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        tmp = np.where(inds == index)[0][0]
        ranks[index] = tmp
        top1[index] = inds[0]
        top5[index] = inds[0:5]
        top10[index] = inds[0:10]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1,top5,top10)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(sims ,return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5))
    top10 = np.zeros((npts, 10))

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        ranks[index] = np.where(inds == index)[0][0]
        top1[index] = inds[0]
        top5[index] = inds[0:5]
        top10[index] = inds[0:10]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, top10)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == '__main__':
    opt = opts_xray.parse_opt()

    print(colored("Evaluating model number: {}".format(opt.model_num),'magenta'))

    if opt.model_type in ['regular_model', 'cat_model', 'tag_model', 'pos_enc_const_model', 'pos_enc_vec_model', 'pos_enc_sin_model']:
        rti,rt = evalrank("../checkpoint/model_%s/model_best.pth.tar" % opt.model_num, view=opt.view,
                                       data_path=opt.data_path, split="test", fold5=False)
    elif opt.model_type in ['double_model', 'pretrained_double_model', 'frozen_double_model']:
        rti, rt = evalrank_double("../checkpoint/model_%s/model_best.pth.tar" % opt.model_num,
                           data_path=opt.data_path, split="test", fold5=False)
    elif opt.model_type == 'light_double_model':
        rti, rt = evalrank_light_double("../checkpoint/model_%s/model_best.pth.tar" % opt.model_num,
                                  data_path=opt.data_path, split="test", fold5=False)
    else:
        raise Exception("Unsupported model type")

    np.save('ranks', rti[0])
    np.save('top1', rti[1])
    np.save('top5', rti[2])
    np.save('top10', rti[3])

    np.save('top1_I2T',rt[1])
    np.save('top5_I2T', rt[2])
    np.save('top10_I2T', rt[3])

    np.savetxt('ranks.txt', rti[0], fmt='%i')
    np.savetxt('top1.txt', rti[1], fmt='%i')
    np.savetxt('top5.txt', rti[2], fmt='%i')
