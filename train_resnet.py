import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import json
import os
import argparse

import utils
from model import model

import itertools
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


### ------------------------------------ Dataloader -------------------------------------- ###
def get_dataloader(dataset, train_dir, val_dir, batchsize):

    if 'CIFAR' in dataset :
        if dataset == 'CIFAR10':
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            nb_cls = 10

        elif dataset == 'CIFAR100':
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            nb_cls = 100

        # transformation of the training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

    elif dataset == 'Clothing1M':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 14

        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

    trainloader = DataLoader(ImageFolder(train_dir, transform_train),
                             batch_size=batchsize,
                             shuffle=True,
                             drop_last=True,
                             num_workers=4,
                             pin_memory=True)

    valloader = DataLoader(ImageFolder(val_dir, transform_val),
                           batch_size=batchsize,
                           shuffle=False,
                           drop_last=False,
                           num_workers=4,
                           pin_memory=True)

    return  trainloader, valloader, nb_cls

### --------------------------------------------------------------------------------------------


### ------------------------------------ Distribution -------------------------------------- ###
def GaussianDist(mu, std, N):

    dist = np.array([np.exp(-((i - mu) / std)**2) for i in range(1, N + 1)])

    return dist / np.sum(dist)

### --------------------------------------------------------------------------------------------


### ------------------------ Test with Nested (iterate all possible K) --------------------- ###
def TestNested(logger, epoch, best_acc, best_k, net_feat, net_lpca, net_cls, valloader, out_dir, mask_feat_dim, dropout):

    net_feat.eval()
    net_lpca.eval()
    net_cls.eval()

    bestTop1 = 0

    true_pred = torch.zeros(len(mask_feat_dim)).cuda()
    nb_sample = 0

    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        feature = net_feat(inputs)
        feature_rec, loss_lpca = net_lpca.forward_eval(feature)
        #outputs = []
# nested where?
       # for i in range(len(mask_feat_dim)) :
            #feature_mask = feature * mask_feat_dim[i]
        #    feature_mask = feature_rec * mask_feat_dim[i]
        #    outputs.append(net_cls(fea:ture_mask).unsqueeze(0))

        #outputs = torch.cat(outputs, dim=0)
        outputs = net_cls(feature_rec)

        _, pred = torch.max(outputs, dim=2)
        targets = targets.unsqueeze(0).expand_as(pred)

        true_pred = true_pred + torch.sum(pred == targets, dim=1).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

    acc, k = torch.max((true_pred / nb_sample - 1e-5 * torch.arange(len(mask_feat_dim)).type_as(true_pred)), dim=0)
    acc, k = acc.item(), k.item()

    msg = '\nNested ... Epoch {:d}, Acc {:.3f} %, K {:d} (Best Acc {:.3f} %)'.format(epoch, acc * 100, k, best_acc * 100)
    #print (msg)
    logger.info(msg)

    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc, acc)
        #print (msg)
        logger.info(msg)
        #print ('Saving Best!!!')
        msg='Saving Best!!!'
        logger.info(msg)
        param = {'feat': net_feat.state_dict(),
                 'lpca': net_lpca.state_dict(),
                 'cls': net_cls.state_dict(),
                 }
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))

        best_acc = acc
        best_k = k

    return best_acc, acc, best_k

### --------------------------------------------------------------------------------------------


### ---------------- Test standard (used for model w/o nested, baseline, dropout) ---------- ###
def TestStandard(logger, epoch, best_acc, best_k, net_feat, net_lpca, net_cls, valloader, out_dir, mask_feat_dim, dropout):

    net_feat.eval()
    net_lpca.eval()
    net_cls.eval()

    bestTop1 = 0

    true_pred = torch.zeros(1).cuda()
    nb_sample = 0

    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        feature = net_feat(inputs)
        #feature = F.dropout(feature, p=dropout, training=False)
        feature_rec, loss_lpca = net_lpca.forward(feature)
        outputs = net_cls(feature_rec)

        _, pred = torch.max(outputs, dim=1)

        true_pred = true_pred + torch.sum(pred == targets).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

    acc = true_pred / nb_sample # accuracy
    acc = acc.item()

    msg = 'Standard ... Epoch {:d}, Acc {:.3f} %, (Best Acc {:.3f} %)'.format(epoch, acc * 100, best_acc * 100)
    #print (msg)
    logger.info(msg)

    # save checkpoint
    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc * 100, acc * 100)
        #print (msg)
        logger.info(msg)
        #print ('Saving Best!!!')
        msg='Saving Best!!!'
        logger.info(msg)
        param = {'feat': net_feat.state_dict(),
                 'lpca': net_lpca.state_dict(),
                 'cls': net_cls.state_dict(),
                 }
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))

        best_acc = acc

    return best_acc, acc, len(mask_feat_dim)

### --------------------------------------------------------------------------------------------

# optimizer1 = stiefel_opti(param_g, opt.lrg) # KPCA
# optimizer2 = torch.optim.Adam(param_e1, lr=opt.lr, weight_decay=0) # or SGD

### -------------------------------------- Training  --------------------------------------- ###
def Train(logger, epoch, optimizer1, optimizer2, net_feat, net_lpca, net_cls, trainloader, criterion, \
        dist, mask_feat_dim, dropout, eta, freeze_bn=False):

    msg = '\nEpoch: {:d}'.format(epoch)
    # print(msg)
    logger.info(msg)

    net_feat.train(freeze_bn = freeze_bn)
    net_lpca.train()
    net_cls.train()

    losses_total = utils.AverageMeter()
    losses_ce = utils.AverageMeter()
    losses_lpca = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for batchIdx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        for optim in optimizer1:
            optim.zero_grad()
        optimizer2.zero_grad()

        feature = net_feat(inputs)

        #if dist is not None:
        #    k = np.random.choice(range(len(mask_feat_dim)), p=dist)
        #    mask_k = mask_feat_dim[k]
        #     feature_masked = feature * mask_k
        #else:
        #    feature_masked = F.dropout(feature, p=dropout, training=True)

        feature_rec, loss_lpca = net_lpca.forward(feature)
        outputs = net_cls(feature_rec)

        loss_ce = criterion(outputs, targets)
        loss_total = loss_ce + eta * loss_lpca

        loss_total.backward()
        for optim in optimizer1:
            optim.step()
        optimizer2.step()

        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        losses_total.update(loss_total.item(), inputs.size()[0])
        losses_ce.update(loss_ce.item(), inputs.size()[0])
        losses_lpca.update(loss_lpca.item(), inputs.size()[0])
        top1.update(acc1[0].item(), inputs.size()[0])
        top5.update(acc5[0].item(), inputs.size()[0])

         msg = 'Loss Total: {:.3f} | Loss Ce: {:.3f} | Loss LPCA: {:.3f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(losses_total.avg, losses_ce.avg, losses_lpca.avg, top1.avg, top5.avg)
        logger.info(msg)
        #utils.progress_bar(batchIdx, len(trainloader), msg)

    return losses_total.avg, losses_ce.avg, losses_lpca.avg, top1.avg, top5.avg
    # also return: losses_ce.avg, losses_lpca.avg, ?

### --------------------------------------------------------------------------------------------


### ------------------------------------ Lr Warm Up  --------------------------------------- ###
def LrWarmUp(logger, warmUpIter, lr, lrg, eta, optimizer1, optimizer2, net_feat, net_lpca, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, freeze_bn) :

    nbIter = 0

    while nbIter < warmUpIter:
        net_feat.train(freeze_bn = freeze_bn)
        net_lpca.train()
        net_cls.train()

        losses_total = utils.AverageMeter()
        losses_ce = utils.AverageMeter()
        losses_lpca = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for batchIdx, (inputs, targets) in enumerate(trainloader):
            nbIter += 1
            if nbIter == warmUpIter:
                break
            lrUpdate1 = nbIter / float(warmUpIter) * lr
            lrUpdate2 = nbIter / float(warmUpIter) * lrg
            for optim in optimizer1:
                for g in optim.param_groups:
                    g['lr'] = lrUpdate1

            for g in optimizer2.param_groups:
                g['lr'] = lrUpdate2

            inputs = inputs.cuda()
            targets = targets.cuda()

            for optim in optimizer1:
                optim.zero_grad()
            optimizer2.zero_grad()

            feature = net_feat(inputs)
            feature_rec, loss_lpca = net_lpca.forward(feature)
            outputs = net_cls(feature_rec)

            loss_ce = criterion(outputs, targets)
            loss_total = loss_ce + eta * loss_lpca

            loss_total.backward()
            for optim in optimizer1:
                optim.step()
            optimizer2.step()

            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
            losses_total.update(loss_total.item(), inputs.size()[0])
            losses_ce.update(loss_ce.item(), inputs.size()[0])
            losses_lpca.update(loss_lpca.item(), inputs.size()[0])
            top1.update(acc1[0].item(), inputs.size()[0])
            top5.update(acc5[0].item(), inputs.size()[0])

            msg = 'Loss_total: {:.3f} | Loss_ce: {:.3f} | Loss_lpca: {:.3f} | Lr : {:.5f} | Lrg : {:.5f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(losses_total.avg, losses_ce.avg, losses_lpca.avg, lrUpdate1, lrUpdate2, top1.avg, top5.avg)
            logger.info(msg)
            # utils.progress_bar(batchIdx, len(trainloader), msg)

### --------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

def main(gpu, arch, dropout, out_dir, dataset, train_dir, val_dir, warmUpIter, lr, lrg, eta, nbEpoch, batchsize, \
        momentum=0.9, weightDecay = 5e-4, lrSchedule = [200, 300], num_pc=50, mu=0, nested=1.0, resumePth=None, freeze_bn=False, pretrained=False):

    best_acc = 0  # best test accuracy
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    trainloader, valloader, nb_cls = get_dataloader(dataset, train_dir, val_dir, batchsize)

    # feature net + classifier net (a linear layer)
    net_feat = model.NetFeat(arch = arch,
                             pretrained = pretrained,
                             dataset = dataset)
    feat_dim = net_feat.feat_dim
    # generate mask
    mask_feat_dim = []
    for i in range(feat_dim):
        tmp = torch.cuda.FloatTensor(1, feat_dim).fill_(0)
        tmp[:, : (i + 1)] = 1
        mask_feat_dim.append(tmp)

    # distribution and test function
    dist = GaussianDist(mu, nested, feat_dim) if nested > 0 else None

    net_lpca = model.LPCA(feat_dim=net_feat.feat_dim,
                          num_pc=num_pc,
                          mask_feat_dim=mask_feat_dim,
                          dist=dist)
    net_cls = model.NetClassifier(feat_dim = net_feat.feat_dim,
                                  nb_cls = nb_cls)

    net_feat.cuda()
    net_lpca.cuda()
    net_cls.cuda()
    #feat_dim = net_feat.feat_dim
    best_k = feat_dim

    Test = TestNested if nested > 0 else TestStandard

    # output dir + loss + optimizer
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    logger = utils.get_logger(out_dir)

    # load model
    if resumePth :
        param = torch.load(resumePth)
        net_feat.load_state_dict(param['feat'])
        msg='Loading feature weight from {}'.format(resumePth)
        logger.info(msg)

        net_lpca.load_state_dict(param['lpca'])
        msg='Loading LPCA weight from {}'.format(resumePth)
        logger.info(msg)

        net_cls.load_state_dict(param['cls'])
        msg='Loading classifier weight from {}'.format(resumePth)
        logger.info(msg)

    criterion = nn.CrossEntropyLoss()

    # Accumulate trainable parameters in 2 groups:
    # 1. W_matrix 2. Network params
    optimizer1 = [torch.optim.SGD(itertools.chain(*[net_feat.parameters()]),
                                                 1e-7,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weightDecay),

                 torch.optim.SGD(itertools.chain(*[net_cls.parameters()]),
                                                 1e-7,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weightDecay)]
    optimizer2 = utils.stiefel_opti(itertools.chain(*[net_lpca.parameters()]), lrg=lrg)

    # learning rate warm up
    LrWarmUp(logger, warmUpIter, lr, lrg, eta, optimizer1, optimizer2, net_feat, net_lpca, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, freeze_bn)

    with torch.no_grad():
        best_acc, acc, best_k = Test(logger, 0, best_acc, best_k, net_feat, net_lpca, net_cls, valloader, out_dir, mask_feat_dim, dropout)

    best_acc, best_k = 0, feat_dim
    for optim in optimizer1:
        for g in optim.param_groups:
            g['lr'] = lr
    for g in optimizer2.param_groups:
        g['lr'] = lrg

    #history = {'trainTop1':[], 'best_acc':[], 'trainTop5':[], 'valTop1':[], 'trainLoss':[], 'best_k':[]}
    history = {'trainTop1':[], 'best_acc':[], 'trainTop5':[], 'valTop1':[], 'trainLoss_total':[], 'trainLoss_ce':[], 'trainLoss_lpca':[], 'best_k':[]}

    #lrScheduler = [MultiStepLR(optim, milestones=lrSchedule, gamma=0.1) for optim in optimizer]
    lrScheduler = [MultiStepLR(optim, milestones=lrSchedule, gamma=0.1) for optim in optimizer1]
    lrgScheduler = MultiStepLR(optimizer2, milestones=lrSchedule, gamma=0.1)

    for epoch in range(nbEpoch):
        trainLoss_total, trainLoss_ce, trainLoss_lpca, trainTop1, trainTop5 = Train(logger, epoch, optimizer1, optimizer2, net_feat, net_lpca, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, eta)

        with torch.no_grad() :
            best_acc, valTop1, best_k = Test(logger, epoch, best_acc, best_k, net_feat, net_lpca, net_cls, valloader, out_dir, mask_feat_dim, dropout)

        history['trainTop1'].append(trainTop1)
        history['trainTop5'].append(trainTop5)
        history['trainLoss_total'].append(trainLoss_total)
        history['trainLoss_ce'].append(trainLoss_ce)
        history['trainLoss_lpca'].append(trainLoss_lpca)
        history['valTop1'].append(valTop1)

        history['best_acc'].append(best_acc)
        history['best_k'].append(best_k)

        with open(os.path.join(out_dir, 'history.json'), 'w') as f :
            json.dump(history, f)

        for lr_schedule in lrScheduler:
            lr_schedule.step()
        #for lrg_schedule in lrgScheduler:
        #    lrg_schedule.step()
        lrgScheduler.step()

    msg = 'mv {} {}'.format(out_dir, '{}_Acc{:.3f}_K{:d}'.format(out_dir, best_acc, best_k))
    print (msg)
    #logger.info(msg)
    os.system(msg)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--train-dir', type=str, default='./data/CIFAR10/train', help='train directory')
    parser.add_argument('--val-dir', type=str, default='./data/CIFAR10/val', help='val directory')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'Clothing1M'], default='Clothing1M', help='which dataset?')

    # training
    parser.add_argument('--warmUpIter', type=int, default=6000, help='total iterations for learning rate warm')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--lrg', default=1e-4, type=float, help='learning rate of LPCA')
    parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--batchsize', type=int, default=320, help='batch size')
    parser.add_argument('--nbEpoch', type=int, default=350, help='nb epoch')
    parser.add_argument('--lrSchedule', nargs='+', type=int, default=[200, 300], help='lr schedule')
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

    # model
    parser.add_argument('--eta', default=0.1, type=float, help='coefficient for the lpca loss')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50'], default='resnet18', help='which archtecture?')
    parser.add_argument('--num-pc', type=int, default=50, help='number of pc in the KPCA block')
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--mu', type=float, default=0.0, help='nested mean hyperparameter')
    parser.add_argument('--nested', type=float, default=0.0, help='nested std hyperparameter')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--resumePth', type=str, help='resume path')
    parser.add_argument('--freeze-bn', action='store_true', help='freeze the BN layers')
    parser.add_argument('--pretrained', action='store_true', help='Start with ImageNet pretrained model (Pytorch Model Zoo)')

    args = parser.parse_args()
    print (args)

    if args.nested > 0 and args.dropout > 0 :
        raise RuntimeError('Activating both nested (eta = {:.3f}) and dropout (ratio = {:.3f})'.format(args.nested, args.dropout))

    main(gpu = args.gpu,

         arch = args.arch,

         dropout = args.dropout,

         out_dir = args.out_dir,

         dataset = args.dataset,

         train_dir = args.train_dir,

         val_dir = args.val_dir,

         warmUpIter = args.warmUpIter,

         lr = args.lr,

         lrg = args.lrg,

         eta = args.eta,

         nbEpoch = args.nbEpoch,

         batchsize = args.batchsize,

         momentum = args.momentum,

         weightDecay = args.weightDecay,

         lrSchedule = args.lrSchedule,

         num_pc = args.num_pc,

         mu = args.mu,

         nested = args.nested,

         resumePth = args.resumePth,

         freeze_bn = args.freeze_bn,

         pretrained = args.pretrained)
