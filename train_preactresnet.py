import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import json
import os
import argparse

import utils
import model.resnet_lpca as resnet

import itertools
import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


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


### ------------------------ Compute mean_feat on whole training set ----------------------- ###
def Statistics(trainloader, net):
    net.eval()
    feat_list = []

    for batchIdx, (inputs, targets) in enumerate(trainloader) :
        inputs = inputs.cuda()
        targets = targets.cuda()

        x_feat_raw = net(inputs)[1]
        feat_list.append(x_feat_raw)

    feat_list = torch.cat(feat_list, dim=0)
    mean_feat = torch.mean(feat_list, dim=0)

    return mean_feat


def GaussianDist(mu, std, N):
    dist = np.array([np.exp(-((i - mu) / std)**2) for i in range(1, N + 1)])

    return dist / np.sum(dist)

### --------------------------------------------------------------------------------------------


### ------------------------ Test with Nested (iterate all possible K) --------------------- ###
def TestNested(logger, epoch, best_acc, best_k, net, valloader, out_dir, mask_feat_dim):

    net.eval()

    bestTop1 = 0

    true_pred = torch.zeros(len(mask_feat_dim)).cuda()
    nb_sample = 0

    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        feature = net(inputs)[2]
        hidden_feat = torch.mm(feature, net.W.t())
        outputs = []

        for i in range(len(mask_feat_dim)):
            x_rec = torch.mm(hidden_feat * mask_feat_dim[i], net.W)
            outputs.append(net.linear(x_rec).unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)

        _, pred = torch.max(outputs, dim=2)
        targets = targets.unsqueeze(0).expand_as(pred)

        true_pred = true_pred + torch.sum(pred == targets, dim=1).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

    acc, k = torch.max((true_pred / nb_sample - 1e-5 * torch.arange(len(mask_feat_dim)).type_as(true_pred)), dim=0)
    acc, k = acc.item(), k.item()

    msg = '\nNested ... Epoch {:d}, Acc {:.3f} %, K {:d} (Best Acc {:.3f} %)'.format(epoch, acc * 100, k, best_acc * 100)
    logger.info(msg)

    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc, acc)
        logger.info(msg)
        logger.info('Saving Best!!!')
        param = {'net': net.state_dict()}
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))

        best_acc = acc
        best_k = k

    return best_acc, acc, best_k

### --------------------------------------------------------------------------------------------


### ---------------- Test standard (used for model w/o nested, baseline, dropout) ---------- ###
def TestStandard(logger, epoch, best_acc, best_k, net, valloader, out_dir, mask_feat_dim):

    net.eval()

    bestTop1 = 0

    true_pred = torch.zeros(1).cuda()
    nb_sample = 0

    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = net(inputs)[0]

        _, pred = torch.max(outputs, dim=1)

        true_pred = true_pred + torch.sum(pred == targets).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

    acc = true_pred / nb_sample # accuracy
    acc = acc.item()

    msg = 'Standard ... Epoch {:d}, Acc {:.3f} %, (Best Acc {:.3f} %)'.format(epoch, acc * 100, best_acc * 100)
    logger.info(msg)

    # save checkpoint
    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc * 100, acc * 100)
        logger.info(msg)
        logger.info('Saving Best!!!')
        param = {'net': net.state_dict()}
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))

        best_acc = acc

    return best_acc, acc, len(mask_feat_dim)

### --------------------------------------------------------------------------------------------


### -------------------------------------- Training  --------------------------------------- ###
def Train(logger, epoch, optimizer1, optimizer2, net, trainloader, criterion, dist, mask_feat_dim, mixup_fn, eta):

    msg = '\nEpoch: {:d}'.format(epoch)
    logger.info(msg)

    net.train()

    losses_total = utils.AverageMeter()
    losses_ce = utils.AverageMeter()
    losses_lpca = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for batchIdx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        if mixup_fn is not None :
            inputs_mixup, targets_mixup = mixup_fn(inputs, targets)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        if dist is not None:
            k = np.random.choice(range(len(mask_feat_dim)), p=dist)
            net.netsed_k = k
            if mixup_fn is not None:
                outputs, _, _, loss_lpca = net(inputs_mixup)
            else:
                outputs, _, _, loss_lpca = net(inputs)
        elif mixup_fn is not None:
            outputs, _, _, loss_lpca = net(inputs_mixup)
        else:
            outputs, _, _, loss_lpca = net(inputs)

        if mixup_fn is not None :
            loss_ce = criterion(outputs, targets_mixup)
        else:
            loss_ce = criterion(outputs, targets)

        loss_total = loss_ce + eta * loss_lpca

        loss_total.backward()
        optimizer1.step()
        optimizer2.step()

        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        if mixup_fn is not None:
            losses_total.update(loss_total.item(), inputs_mixup.size()[0])
            losses_ce.update(loss_ce.item(), inputs_mixup.size()[0])
            losses_lpca.update(loss_lpca.item(), inputs_mixup.size()[0])
            top1.update(acc1[0].item(), inputs_mixup.size()[0])
            top5.update(acc5[0].item(), inputs_mixup.size()[0])
        else:
            losses_total.update(loss_total.item(), inputs.size()[0])
            losses_ce.update(loss_ce.item(), inputs.size()[0])
            losses_lpca.update(loss_lpca.item(), inputs.size()[0])
            top1.update(acc1[0].item(), inputs.size()[0])
            top5.update(acc5[0].item(), inputs.size()[0])


        if batchIdx % 50 == 0 :
            msg = 'batchIdx: {:d}/{:d} | Loss Total: {:.3f} | Loss Ce: {:.3f} | Loss LPCA: {:.3f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(batchIdx, len(trainloader), losses_total.avg, losses_ce.avg, losses_lpca.avg, top1.avg, top5.avg)
            logger.info(msg)

    return losses_total.avg, losses_ce.avg, losses_lpca.avg, top1.avg, top5.avg

### --------------------------------------------------------------------------------------------


### ------------------------------------ Lr Warm Up  --------------------------------------- ###
def LrWarmUp(logger, warmUpIter, lr, lrg, eta, optimizer1, optimizer2, net, trainloader, criterion, dist, mask_feat_dim, mixup_fn) :

    nbIter = 0

    while nbIter < warmUpIter:
        net.train()

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
            for g in optimizer1.param_groups:
                g['lr'] = lrUpdate1

            for g in optimizer2.param_groups:
                g['lr'] = lrUpdate2

            inputs = inputs.cuda()
            targets = targets.cuda()
            
            if mixup_fn is not None:
                inputs_mixup, targets_mixup = mixup_fn(inputs, targets)
                
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if dist is not None:
                k = np.random.choice(range(len(mask_feat_dim)), p=dist)
                net.netsed_k = k
                if mixup_fn is not None:
                    outputs, _, _, loss_lpca = net(inputs_mixup)
                else:
                    outputs, _, _, loss_lpca = net(inputs)
            elif mixup_fn is not None:
                outputs, _, _, loss_lpca = net(inputs_mixup)
            else:
                outputs, _, _, loss_lpca = net(inputs)

            if mixup_fn is not None:
                loss_ce = criterion(outputs, targets_mixup)
            else:
                loss_ce = criterion(outputs, targets)

            loss_ce = criterion(outputs, targets)
            loss_total = loss_ce + eta * loss_lpca

            loss_total.backward()
            optimizer1.step()
            optimizer2.step()

            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
            if mixup_fn is not None:
                losses_total.update(loss_total.item(), inputs_mixup.size()[0])
                losses_ce.update(loss_ce.item(), inputs_mixup.size()[0])
                losses_lpca.update(loss_lpca.item(), inputs_mixup.size()[0])
                top1.update(acc1[0].item(), inputs_mixup.size()[0])
                top5.update(acc5[0].item(), inputs_mixup.size()[0])
            else:
                losses_total.update(loss_total.item(), inputs.size()[0])
                losses_ce.update(loss_ce.item(), inputs.size()[0])
                losses_lpca.update(loss_lpca.item(), inputs.size()[0])
                top1.update(acc1[0].item(), inputs.size()[0])
                top5.update(acc5[0].item(), inputs.size()[0])


            if nbIter % 200 == 0 :
                msg = 'Training Iter: {:d} / {:d} | Loss_total: {:.3f} | Loss_ce: {:.3f} | Loss_lpca: {:.3f} | Lr : {:.5f} | Lrg : {:.5f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(nbIter, warmUpIter, losses_total.avg, losses_ce.avg, losses_lpca.avg, lrUpdate1, lrUpdate2, top1.avg, top5.avg)
                logger.info(msg)

### --------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

def main(gpu, arch, out_dir, dataset, train_dir, val_dir, warmUpIter, lr, lrg, eta, nbEpoch, batchsize, \
        momentum=0.9, weightDecay = 5e-4, lrSchedule = [200, 300], num_pc=50, mu=0, nested=1.0, mixup = 0.0, resumePth=None):

    best_acc = 0  # best test accuracy
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    trainloader, valloader, nb_cls = get_dataloader(dataset, train_dir, val_dir, batchsize)

    # define net
    if arch == "PreActResNet18":
        net = resnet.PreActResNet18(h_dim=num_pc, num_classes=nb_cls)
        net.cuda()

    mask_feat_dim = []
    for i in range(num_pc):
        tmp = torch.cuda.FloatTensor(1, num_pc).fill_(0)
        tmp[:, : (i + 1)] = 1
        mask_feat_dim.append(tmp)

    # distribution and test function
    dist = GaussianDist(mu, nested, num_pc) if nested > 0 else None
    best_k = num_pc

    Test = TestNested if nested > 0 else TestStandard

    # output dir + loss + optimizer
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    logger = utils.get_logger(out_dir)

    # load model
    if resumePth:
        param = torch.load(resumePth)
        net.load_state_dict(param['net'])
        msg = 'Loading net weight from {}'.format(resumePth)
        logger.info(msg)

    mixup_fn = None
    mixup_active = args.mixup > 0
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=args.mixup, prob=1, num_classes=nb_cls)

    # criterion for supervised learning loss
    if mixup_fn is not None :
        criterion = SoftTargetCrossEntropy().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss()

    # Accumulate trainable parameters in 2 groups:
    # 1. U_matrix 2. Network params
    param_g, param_el = utils.param_state(net)

    # optimization over the classifier
    optimizer1 = torch.optim.SGD(param_el,
                                 lr = 1e-7,
                                 momentum = momentum,
                                 weight_decay = weightDecay)
    optimizer2 = utils.stiefel_opti(param_g, lrg=lrg)

    # learning rate warm up
    LrWarmUp(logger, warmUpIter, lr, lrg, eta, optimizer1, optimizer2, net, trainloader, criterion, dist, mask_feat_dim, mixup_fn)

    with torch.no_grad():
        # we update the mean_feat
        mean_feat = Statistics(trainloader, net)
        net.mean_feat = mean_feat
        best_acc, acc, best_k = Test(logger, 0, best_acc, best_k, net, valloader, out_dir, mask_feat_dim)

    best_acc, best_k = 0, num_pc
    for g in optimizer1.param_groups:
        g['lr'] = lr
    for g in optimizer2.param_groups:
        g['lr'] = lrg

    history = {'trainTop1':[], 'best_acc':[], 'trainTop5':[], 'valTop1':[], 'trainLoss_total':[], 'trainLoss_ce':[], 'trainLoss_lpca':[], 'best_k':[]}

    lrScheduler = MultiStepLR(optimizer1, milestones=lrSchedule, gamma=0.1)
    lrgScheduler = MultiStepLR(optimizer2, milestones=lrSchedule, gamma=0.2)

    for epoch in range(nbEpoch):
        trainLoss_total, trainLoss_ce, trainLoss_lpca, trainTop1, trainTop5 = Train(logger, epoch, optimizer1, optimizer2, net, trainloader, criterion, dist, mask_feat_dim,  mixup_fn, eta)

        with torch.no_grad() :
            # we update the mean_feat
            mean_feat = Statistics(trainloader, net)
            net.mean_feat = mean_feat
            best_acc, valTop1, best_k = Test(logger, epoch, best_acc, best_k, net, valloader, out_dir, mask_feat_dim)

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

        lrScheduler.step()
        lrgScheduler.step()

    # save last model
    logger.info('Saving Last!!!')
    param = {'net': net.state_dict()}
    torch.save(param, os.path.join(out_dir, 'netLast.pth'))

    msg = 'mv {} {}'.format(out_dir, '{}_Acc{:.3f}_K{:d}'.format(out_dir, best_acc, best_k))
    logger.info(msg)
    os.system(msg)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--train-dir', type=str, default='./data/CIFAR10/train', help='train directory')
    parser.add_argument('--val-dir', type=str, default='./data/CIFAR10/val', help='val directory')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', help='which dataset?')

    # training
    parser.add_argument('--warmUpIter', type=int, default=6000, help='total iterations for learning rate warm')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--lrg', default=1e-4, type=float, help='learning rate of LPCA')
    parser.add_argument('--weightDecay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--nbEpoch', type=int, default=200, help='nb epoch')
    parser.add_argument('--lrSchedule', nargs='+', type=int, default=[100, 150], help='lr schedule')
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

    # model
    parser.add_argument('--eta', default=0.1, type=float, help='coefficient for the lpca loss')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50', 'PreActResNet18'], default='PreActResNet18', help='which archtecture?')
    parser.add_argument('--num-pc', type=int, default=50, help='number of pc in the KPCA block')
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--mu', type=float, default=0.0, help='nested mean hyperparameter')
    parser.add_argument('--nested', type=float, default=0.0, help='nested std hyperparameter')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--resumePth', type=str, help='resume path')

    args = parser.parse_args()
    print (args)

    main(gpu = args.gpu,

         arch = args.arch,

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

         mixup = args.mixup,

         resumePth = args.resumePth)
