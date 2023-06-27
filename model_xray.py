"""SGRAF model"""
import copy

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict

FRONTAL_TAG = 0.05
LATERAL_TAG = -0.05

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        # print("EdanMayan EncoderImage sizes: img_dim: ", img_dim, " embed_size: ", embed_size) - debug
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations .
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """

    def __init__(self, word_dim, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.fc = nn.Linear(word_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, captions):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        cap_emb = self.fc(captions)

        return cap_emb




class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global, weights


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        #return sim_saf, sim_attn
        return sim_saf

    # I added return sim_attn
    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, sim_dim, module_name='AVE', sgr_step=3, model_type='regular_model'):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name
        num_image_features = 49*(2 if (model_type in ['cat_model', 'tag_model', 'pos_enc_const_model',
                                                      'pos_enc_vec_model', 'pos_enc_sin_model']) else 1)
        #self.v_global_w = VisualSA(embed_size, 0.4, 100)
        self.v_global_w = VisualSA(embed_size, 0.4, num_image_features)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for i in range(sgr_step)])
        elif module_name == 'SAF':
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError('Invalid input of opt.module_name in opts.py')

        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        # my additon
        sim_attn_all = []
        cap_weights_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i, cap_weights = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == 'SGR':
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)
            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)

        sim_all = torch.cat(sim_all, 1)
        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn * smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class simCLRloss(nn.Module):
    """
    Compute simCLR loss
    """

    def __init__(self, temp1):
        super(simCLRloss, self).__init__()
        self.temp = float(temp1)

    def forward(self, scores):
        batch_size = scores.shape[0]
        labels = Variable(torch.LongTensor(range(batch_size))).to(scores.device)
        scores1 = scores.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores / self.temp, labels)
        loss1 = nn.CrossEntropyLoss()(scores1 / self.temp, labels)
        total_loss = loss0 + loss1
        return total_loss

class lossSum(nn.Module):
    """
    Compute simCLR and contrastive loss sum
    """

    def __init__(self, temp1, margin, max_violation):
        super(lossSum, self).__init__()
        self.simCLR = simCLRloss(temp1=temp1)
        self.ContrastiveLoss = ContrastiveLoss(margin=margin, max_violation=max_violation)

    def forward(self, scores):
        total_loss = 2*self.simCLR(scores) + self.ContrastiveLoss(scores)
        return total_loss

class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.word_dim, opt.embed_size,
                                   no_txtnorm=opt.no_txtnorm)
        if hasattr(opt, 'model_type'):
            self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                             opt.module_name, opt.sgr_step, opt.model_type)
            self.model_type = opt.model_type
        else:
            self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)
            self.model_type = 'regular_model'

        self.batch_size = opt.batch_size

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss == 'ContrastiveLoss':
            self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        elif opt.loss == 'simCLRloss':
            self.criterion = simCLRloss(temp1=opt.temp)

        elif opt.loss == 'lossSum':
            self.criterion = lossSum(temp1=opt.temp, margin=opt.margin, max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        if self.model_type == 'tag_model':
            tag = torch.ones(images.shape[0], images.shape[1], 1).cuda()
            tag[:, :int(images.shape[1] / 2), :] *= FRONTAL_TAG
            tag[:, int(images.shape[1] / 2):, :] *= LATERAL_TAG
            images = torch.cat((images, tag), dim=2)

        if self.model_type == 'pos_enc_sin_model':
            _ , num_vecs, img_dim = images.shape
            P = torch.zeros((1, num_vecs, img_dim))
            X = torch.arange(0, num_vecs, dtype=torch.float32).reshape(-1, 1)
            X = X / torch.pow(10_000, torch.arange(0, img_dim, 2, dtype=torch.float32) / img_dim)
            P[:, :, 0::2] = 0.5*torch.sin(X)
            P[:, :, 1::2] = 0.5*torch.cos(X)
            images = images + P.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()


class DoubleSGRAF(object):
    """
    Averaging two Similarity Reasoning and Filtration (SGRAF) Networks
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.img_encF = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_encF = EncoderText(opt.word_dim, opt.embed_size,
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_encF = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)

        self.img_encL = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_encL = EncoderText(opt.word_dim, opt.embed_size,
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_encL = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)

        self.batch_size = opt.batch_size

        self.mean_layer = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        if torch.cuda.is_available():
            self.img_encF.cuda()
            self.txt_encF.cuda()
            self.sim_encF.cuda()
            self.img_encL.cuda()
            self.txt_encL.cuda()
            self.sim_encL.cuda()
            self.mean_layer.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss == 'ContrastiveLoss':
            self.criterion = ContrastiveLoss(margin=opt.margin,
                                             max_violation=opt.max_violation)
        elif opt.loss == "simCLRloss":
            self.criterion = simCLRloss(temp1=opt.temp)

        elif opt.loss == 'lossSum':
            self.criterion = lossSum(temp1=opt.temp, margin=opt.margin, max_violation=opt.max_violation)

        self.Eiters = 0

        if opt.model_type in ['pretrained_double_model', 'frozen_double_model']:
            initial_state_dict = self.state_dict()

            model_path_frontal = "../checkpoint/model_13/model_best.pth.tar"
            initial_state_dict[:3] = torch.load(model_path_frontal)['model']

            model_path_lateral = "../checkpoint/model_14/model_best.pth.tar"
            initial_state_dict[3:6] = torch.load(model_path_lateral)['model']

            with torch.no_grad():
                self.load_state_dict(initial_state_dict)
                self.mean_layer.weight = torch.nn.Parameter(torch.tensor([[[[0.5]],[[0.5]]]]).cuda())
                self.mean_layer.bias = torch.nn.Parameter(torch.tensor([0.0]).cuda())

        if opt.model_type == 'frozen_double_model':
            params = list(self.mean_layer.parameters())
        else:
            params = list(self.txt_encF.parameters())
            params += list(self.img_encF.parameters())
            params += list(self.sim_encF.parameters())
            params += list(self.txt_encL.parameters())
            params += list(self.img_encL.parameters())
            params += list(self.sim_encL.parameters())
            params += list(self.mean_layer.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    def state_dict(self):
        state_dict = [self.img_encF.state_dict(), self.txt_encF.state_dict(), self.sim_encF.state_dict(),
                      self.img_encL.state_dict(), self.txt_encL.state_dict(), self.sim_encL.state_dict(),
                      self.mean_layer.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_encF.load_state_dict(state_dict[0])
        self.txt_encF.load_state_dict(state_dict[1])
        self.sim_encF.load_state_dict(state_dict[2])
        self.img_encL.load_state_dict(state_dict[3])
        self.txt_encL.load_state_dict(state_dict[4])
        self.sim_encL.load_state_dict(state_dict[5])
        self.mean_layer.load_state_dict(state_dict[6])

    def train_start(self):
        """switch to train mode"""
        self.img_encF.train()
        self.txt_encF.train()
        self.sim_encF.train()
        self.img_encL.train()
        self.txt_encL.train()
        self.sim_encL.train()
        self.mean_layer.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_encF.eval()
        self.txt_encF.eval()
        self.sim_encF.eval()
        self.img_encL.eval()
        self.txt_encL.eval()
        self.sim_encL.eval()
        self.mean_layer.eval()

    def forward_emb(self, imagesF, imagesL, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            imagesF = imagesF.cuda()
            imagesL = imagesL.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        # print("EdanMayan3 iamges sizes: ", images.shape) - debug
        img_embsF = self.img_encF(imagesF)
        cap_embsF = self.txt_encF(captions)
        img_embsL = self.img_encL(imagesL)
        cap_embsL = self.txt_encL(captions)
        return img_embsF, cap_embsF, img_embsL, cap_embsL, lengths

    def forward_sim(self, img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens):
        # Forward similarity encoding
        simsF = self.sim_encF(img_embsF, cap_embsF, cap_lens)
        simsL = self.sim_encL(img_embsL, cap_embsL, cap_lens)
        simsF = torch.unsqueeze(simsF, dim=0)
        simsL = torch.unsqueeze(simsL, dim=0)
        sims_2D = torch.cat([simsF,simsL])
        sims = self.mean_layer(sims_2D)
        return sims[0]

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, imagesF, imagesL, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens = self.forward_emb(imagesF, imagesL, captions, lengths)
        sims = self.forward_sim(img_embsF, cap_embsF, img_embsL, cap_embsL, cap_lens)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()


class lightDoubleSGRAF(object):
    """
    Averaging two Similarity Reasoning and Filtration (SGRAF) Networks with shared caps
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.img_encF = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.sim_encF = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)

        self.img_encL = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.sim_encL = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)

        self.txt_enc = EncoderText(opt.word_dim, opt.embed_size,
                                   no_txtnorm=opt.no_txtnorm)

        self.batch_size = opt.batch_size

        self.mean_layer = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        if torch.cuda.is_available():
            self.img_encF.cuda()
            self.sim_encF.cuda()
            self.img_encL.cuda()
            self.sim_encL.cuda()
            self.txt_enc.cuda()
            self.mean_layer.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss == 'ContrastiveLoss':
            self.criterion = ContrastiveLoss(margin=opt.margin,
                                             max_violation=opt.max_violation)
        elif opt.loss == "simCLRloss":
            self.criterion = simCLRloss(temp1=opt.temp)

        elif opt.loss == 'lossSum':
            self.criterion = lossSum(temp1=opt.temp, margin=opt.margin, max_violation=opt.max_violation)


        params = list(self.img_encF.parameters())
        params += list(self.sim_encF.parameters())
        params += list(self.img_encL.parameters())
        params += list(self.sim_encL.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.mean_layer.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_encF.state_dict(),  self.sim_encF.state_dict(), self.img_encL.state_dict(),
                      self.sim_encL.state_dict(), self.txt_enc.state_dict(), self.mean_layer.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_encF.load_state_dict(state_dict[0])
        self.sim_encF.load_state_dict(state_dict[1])
        self.img_encL.load_state_dict(state_dict[2])
        self.sim_encL.load_state_dict(state_dict[3])
        self.txt_enc.load_state_dict(state_dict[4])
        self.mean_layer.load_state_dict(state_dict[5])

    def train_start(self):
        """switch to train mode"""
        self.img_encF.train()
        self.sim_encF.train()
        self.img_encL.train()
        self.sim_encL.train()
        self.txt_enc.train()
        self.mean_layer.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_encF.eval()
        self.sim_encF.eval()
        self.img_encL.eval()
        self.sim_encL.eval()
        self.txt_enc.eval()
        self.mean_layer.eval()

    def forward_emb(self, imagesF, imagesL, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            imagesF = imagesF.cuda()
            imagesL = imagesL.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        # print("EdanMayan3 iamges sizes: ", images.shape) - debug
        img_embsF = self.img_encF(imagesF)
        img_embsL = self.img_encL(imagesL)
        cap_embs = self.txt_enc(captions)
        return img_embsF, img_embsL, cap_embs, lengths

    def forward_sim(self, img_embsF, img_embsL, cap_embs, cap_lens):
        # Forward similarity encoding
        simsF = self.sim_encF(img_embsF, cap_embs, cap_lens)
        simsL = self.sim_encL(img_embsL, cap_embs, cap_lens)
        simsF = torch.unsqueeze(simsF, dim=0)
        simsL = torch.unsqueeze(simsL, dim=0)
        sims_2D = torch.cat([simsF,simsL])
        sims = self.mean_layer(sims_2D)
        return sims[0]

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, imagesF, imagesL, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embsF, img_embsL, cap_embs, cap_lens = self.forward_emb(imagesF, imagesL, captions, lengths)
        sims = self.forward_sim(img_embsF, img_embsL, cap_embs, cap_lens)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
