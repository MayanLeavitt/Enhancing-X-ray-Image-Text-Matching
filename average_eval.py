import time

import torch
import numpy as np

from evaluation_xray import i2t, t2i, shard_attn_scores, encode_data

from model_xray import SGRAF

from data_xray import get_test_loader

import opts_xray


def evalrank_avg(model_path1, model_path2, data_path=None, mean="regular", split='test', fold5=False):
    """
    Evaluate a 2 trained models on either dev or test by averaging the similarity scores. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    First model is trained with frontal images and second model is trained with lateral images.
    """
    # load model and options
    checkpoint1 = torch.load(model_path1)
    checkpoint2 = torch.load(model_path2)
    opt1 = checkpoint1['opt']
    opt2 = checkpoint2['opt']
    save_epoch1 = checkpoint1['epoch']
    save_epoch2 = checkpoint2['epoch']
    print(model_path1, ": ", opt1)
    print(model_path2, ": ", opt2)
    if data_path is not None:
        opt.data_path = data_path


    # construct model
    model1 = SGRAF(opt1)
    model2 = SGRAF(opt2)

    # load model state
    model1.load_state_dict(checkpoint1['model'])
    model2.load_state_dict(checkpoint2['model'])


    print('Loading dataset')
    data_loader1 = get_test_loader(split, opt1.data_name,
                                  opt1.batch_size, opt1.workers, opt1, "frontal")
    data_loader2 = get_test_loader(split, opt2.data_name,
                                  opt2.batch_size, opt2.workers, opt2, "lateral")

    print("=> loaded checkpoint_epoch {}".format(save_epoch1))
    print("=> loaded checkpoint_epoch {}".format(save_epoch2))
    #
    print('Computing results...')
    img_embs1, cap_embs1, cap_lens1 = encode_data(model1, data_loader1)
    img_embs2, cap_embs2, cap_lens2 = encode_data(model2, data_loader2)


    print('Frontal Images: %d, Captions: %d' %
          (img_embs1.shape[0], cap_embs1.shape[0]))
    print('Lateral Images: %d, Captions: %d' %
          (img_embs2.shape[0], cap_embs2.shape[0]))

    # record computation time of validation
    start = time.time()
    # sims is the similarity score (output of the model)
    print("Frontal: ")
    sims1 = shard_attn_scores(model1, img_embs1, cap_embs1, cap_lens1, opt1, shard_size=100)
    print("Lateral: ")
    sims2 = shard_attn_scores(model2, img_embs2, cap_embs2, cap_lens2, opt2, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end - start)

    if mean == 'regular':
        sims_mean = (sims1+sims2)/2
    else:
        r1, rt1 = i2t(sims1, return_ranks=True)
        ri1, rti1 = t2i(sims1, return_ranks=True)
        rsum1 = r1[0] + r1[1] + r1[2] + ri1[0] + ri1[1] + ri1[2]
        r2, rt2 = i2t(sims2, return_ranks=True)
        ri2, rti2 = t2i(sims2, return_ranks=True)
        rsum2 = r2[0] + r2[1] + r2[2] + ri2[0] + ri2[1] + ri2[2]
        w1 = rsum1/(rsum1+rsum2)
        w2 = rsum2/(rsum1+rsum2)
        sims_mean = sims1*w1 + sims2*w2

    # bi-directional retrieval
    r, rt = i2t(sims_mean, return_ranks=True)
    ri, rti = t2i(sims_mean, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    return rti,rt

if __name__ == '__main__':
    opt = opts_xray.parse_opt()
    rti,rt = evalrank_avg("../checkpoint/model_19/model_best.pth.tar",
                          "../checkpoint/model_20/model_best.pth.tar",
                                       data_path=opt.data_path, mean=opt.average_eval, split="test", fold5=False)

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