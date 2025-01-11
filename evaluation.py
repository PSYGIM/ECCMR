import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import DataLoader
from model import *
from dataloading import *
import numpy as np
import torch.optim as optimizer
import scipy.io as sio
from param import args
import argparse
import time
from param import parse_args

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 留下seq_k等于0的坐标
    return pad_attn_mask.expand(batch_size, len_q, len_k)
def load_data():
    opt = parse_args()
    val_loc_img = np.load('H:/PengShouyong/ImageTextMatching_2/ptp-main/src/blip_src/val_loc_img_zc.npy',allow_pickle=True)
    val_loc_txt = np.load('H:/PengShouyong/ImageTextMatching_2/ptp-main/src/blip_src/val_loc_txt_zc.npy',allow_pickle=True)
    # val_img_atts = np.load('H:/PengShouyong/ImageTextMatching_2/ptp-main/src/blip_src/val_img_atts.npy',allow_pickle=True)
    val_txt_atts = np.load('H:/PengShouyong/ImageTextMatching_2/ptp-main/src/blip_src/val_txt_atts.npy',allow_pickle=True)
    val_loc_img = np.array([row for row in val_loc_img for _ in range(5)])
    # val_image = np.array([row for row in val_image for _ in range(5)])
    val_txt_atts = val_txt_atts[:, 1 :]

    val_loader = DataLoader(MyDataSet(val_loc_img,val_loc_txt,val_txt_atts),
                    batch_size=opt.batch_size,
                    shuffle=False)
    return  val_loader



from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train(model, loader,eva_loader, optimizer,accumulation_steps):
    opt = parse_args()
    scaler = GradScaler()
    for epoch in range(0, opt.num_epochs):
        model.train()
        print('epoch:', epoch)
        total_loss = 0.0
        num_batches = 0

        for i,(image,text,txt_attn) in enumerate(tqdm(loader)):

            image_atts =torch.ones(image.size()[:-1], dtype=torch.long).cuda()
            image, text, text_atts= image.cuda(),text.cuda(), txt_attn.cuda()

            with autocast():
                loss = model(image, image_atts, text, text_atts)
            scaler.scale(loss).backward()
            if (num_batches + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        print(total_loss)
        if (epoch+1) % 1 == 0:
                score = evaluation(model, eva_loader)
                torch.save(model.state_dict(), "./transformer.pth")


def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size + 1)
    n_cap_shard = int((len(captions) - 1) / shard_size + 1)

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_t2i(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_i2t(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d
def i2t(images, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def evaluation(model, loader):
    # model.eval()
    opt = parse_args()
    cap_embs=[]
    img_embs=[]
    row_sums = loader.dataset.txt_attn.sum(axis=1)
    cap_lens = [np.array(x) for x in row_sums]
    # loader.dataset.loc_txt, loader.dataset.txt_attn, loader.dataset.loc_img = \
    #     torch.tensor(loader.dataset.loc_txt).cuda(),torch.tensor(loader.dataset.txt_attn).cuda(), torch.tensor(loader.dataset.loc_img).cuda()
    with torch.no_grad():
        for (loc_img,loc_txt,txt_attn) in tqdm(loader):
            image_atts = torch.ones(loc_img.shape[:-1], dtype=torch.long).cuda()
            loc_txt,txt_attn,loc_img = loc_txt.cuda(),txt_attn.cuda(), loc_img.cuda()
            cap_embs_c,img_embs_c = model.cross_att(loc_txt,txt_attn,loc_img,image_atts)
            cap_emb, img_emb = model.self_att(cap_embs_c,txt_attn,img_embs_c,image_atts)
            lang_feats_c, visn_feats_c = model.output_fc(cap_emb,img_emb)
            cap_embs.append(lang_feats_c)
            img_embs.append(visn_feats_c)
    start = time.time()
    cap_embs = torch.cat(cap_embs, dim=0).cpu().numpy()
    img_embs = torch.cat(img_embs, dim=0).cpu().numpy()
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    else:
        raise NotImplementedError
    end = time.time()
    print("calculate similarity time:", end - start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    print('i2t : r1', r1,'r5', r5, 'r10',r10,'t2i : r1', r1i,'r5', r5i,'r10', r10i,'rsum', currscore,)

    return currscore


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    x1 = torch.sum(x1 * x2, dim)
    x2 = torch.norm(x2, 2, dim)
    return (x1 / (x2).clamp(min=eps)).squeeze()
def intra_relation(K, Q, xlambda):
    """
    Q: (n_context, sourceL, d)
    K: (n_context, sourceL, d)
    return (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)

    attn = attn.view(batch_size*KL, KL)
    attn = nn.Softmax(dim=1)(attn*xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn
def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

        # --> (batch*sourceL, queryL)
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)
    # --> (batch, sourceL, queryL)
    attn = attn.view(batch_size, sourceL, queryL)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    # attn --> (batch, sourceL, queryL)
    # positive attention
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * lamda)
    attn_pos = l1norm(attn_pos, 1)
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)

    return  attn_pos
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
def main():
    print("Creating retrieval dataset")
    eva_loader = load_data()

    model = LXRTXLayer.from_pretrained('bert-base-uncased').cuda()
    state_dict = torch.load('H:/PengShouyong/ImageTextMatching_2/transformertoimagetext_coco test/transformer_coco.pth')
    model.load_state_dict(state_dict)


    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    score = evaluation(model, eva_loader)



    # model = LXRTXLayer.from_pretrained('bert-base-uncased').cuda()
    #
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, weight_decay=0.05)
    # train(model,train_loader,eva_loader,optimizer,accumulation_steps=100)
    # score_val_i2t, score_val_t2i, = evaluation(model, eva_loader)
    # val_result = itm_eval(score_val_i2t, score_val_t2i, eva_loader.dataset.img2txt,eva_loader.dataset.idxs)
    # print(val_result)


if __name__ == '__main__':
    # Hyper Parameters

    main()