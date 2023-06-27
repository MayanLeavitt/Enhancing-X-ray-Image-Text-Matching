"""
# Pytorch implementation for AAAI2021 paper from
# https://arxiv.org/pdf/2101.01368.
# "Similarity Reasoning and Filtration for Image-Text Matching"
# Haiwen Diao, Ying Zhang, Lin Ma, Huchuan Lu
#
# Writen by Haiwen Diao, 2020
"""

import os
import time
import shutil

import torch
import numpy

import data_xray
import opts_xray

from model_xray import SGRAF, DoubleSGRAF, lightDoubleSGRAF
from evaluation_xray import i2t, t2i, AverageMeter, LogCollector, encode_data, encode_data_double, shard_attn_scores,\
                            shard_attn_scores_double, encode_data_light_double, shard_attn_scores_light_double

import logging
import tensorboard_logger as tb_logger


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    opt = opts_xray.parse_opt()

    if (opt.model_type == 'tag_model') and (opt.img_dim != 513):
        raise Exception("Image dimension isn't compatible with tag_model")

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load data loaders
    train_loader, val_loader = data_xray.get_loaders(opt.data_name, opt.batch_size, opt.workers, opt.model_type, opt)
    # Construct the model
    if opt.model_type in ['regular_model','cat_model', 'tag_model', 'pos_enc_const_model', 'pos_enc_vec_model', 'pos_enc_sin_model']:
        model = SGRAF(opt)
    elif opt.model_type in ['double_model', 'pretrained_double_model', 'frozen_double_model']:
        model = DoubleSGRAF(opt)
    elif opt.model_type == 'light_double_model':
        model = lightDoubleSGRAF(opt)
    else:
        raise Exception("Unsupported model type")
    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        r_sum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = r_sum > best_rsum
        best_rsum = max(r_sum, best_rsum)

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):

        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    if opt.model_type in ['regular_model', 'cat_model', 'tag_model', 'pos_enc_const_model', 'pos_enc_vec_model', 'pos_enc_sin_model']:
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
        end = time.time()
        print("calculate similarity time:", end - start)

    elif opt.model_type in ['double_model', 'pretrained_double_model', 'frozen_double_model']:
        img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens = encode_data_double(model, val_loader, opt.log_step, logging.info)

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores_double(model, img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens, opt, shard_size=100)
        end = time.time()
        print("calculate similarity time:", end - start)

    elif opt.model_type == 'light_double_model':
        img_embsF, img_embsL, cap_embs, cap_lens = encode_data_light_double(model, val_loader, opt.log_step, logging.info)

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores_light_double(model, img_embsF, img_embsL, cap_embs, cap_lens, opt, shard_size=100)
        end = time.time()
        print("calculate similarity time:", end - start)


    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r_sum', r_sum, step=model.Eiters)

    return r_sum

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    #lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update)) # for earlier models
    lr = opt.learning_rate * (opt.lr_change ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
