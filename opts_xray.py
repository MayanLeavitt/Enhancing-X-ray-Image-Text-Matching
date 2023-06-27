"""Argument parser"""

import argparse


def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='../data/new_data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='mimiccxr_precomp',
                        help='{coco,f30k,mimiccxr}_precomp')
    parser.add_argument('--vocab_path', default='/apdcephfs/share_1313228/home/haiwendiao/SGRAF-master/vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='../checkpoint/model',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='../log/model',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--view', default='frontal',
                        help='frontal or lateral images')
    parser.add_argument('--model_num', default='0',
                        help='model number')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--temp', default=0.1, type=float,
                        help='temperature of simCLR loss')
    parser.add_argument('--lr_change', default=0.1, type=float,
                        help='size of learning rate change')

    # ------------------------- model setting -----------------------#
    #parser.add_argument('--img_dim', default=2048, type=int,
    #                    help='Dimensionality of the image embedding.')
    parser.add_argument('--img_dim', default=512, type=int,
                         help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=768, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SAF', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')
    parser.add_argument('--loss', default='simCLRloss',
                        help='ContrastiveLoss or simCLRloss loss or lossSum')
    parser.add_argument('--average_eval', default='regular',
                        help='regular mean or weighted mean')
    parser.add_argument('--model_type', default='regular_model',
                        help='regular_model or double_model or cat_model or light_double_model or tag_model'
                             'or pos_enc_const_model or pos_enc_vec_model or pretrained_double_model'
                             'or pos_enc_sin_model or frozen_double_model')

    opt = parser.parse_args()
    print(opt)
    return opt
