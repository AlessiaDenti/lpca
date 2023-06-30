import torch
import torch.nn as nn

import json
import os
import argparse

import utils 
import model.resnet_lpca as resnet

import itertools
import numpy as np 

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


### ------------------------------------ Dataloader -------------------------------------- ### 
def get_dataloader(dataset, test_dir, batchsize): 
    
    if 'CIFAR' in dataset : 
        if dataset == 'CIFAR10': 
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            nb_cls = 10
            
        elif dataset == 'CIFAR100':
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            nb_cls = 100
               
    # transformation of the test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    testloader = DataLoader(ImageFolder(test_dir, transform_test),
                            batch_size=batchsize, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers = 4, 
                            pin_memory = True)        
                                
    return  testloader, nb_cls                    

### -------------------------------------------------------------------------------------------- 


### ------------------------ Compute mean_feat on whole training set ----------------------- ###  
def Statistics(testloader, net):
    net.eval()
    feat_list = []

    for batchIdx, (inputs, targets) in enumerate(testloader) :
        inputs = inputs.cuda()
        targets = targets.cuda()

        x_feat_raw = net(inputs)[1]
        feat_list.append(x_feat_raw)

    feat_list = torch.cat(feat_list, dim=0)
    mean_feat = torch.mean(feat_list, dim=0)

    return mean_feat

### --------------------------------------------------------------------------------------------


### --------------------------- Test with nested (iterate all possible K) ------------------ ###  
def TestNested(best_k, net, testloader, mask_feat_dim):
    
    net.eval()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()    
    
    for batchIdx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda() 
        targets = targets.cuda()
           
        feature = net(inputs)[2]
        hidden_feat = torch.mm(feature, net.W.t()) 

        if best_k is not None:
            hidden_feat_rec = torch.mm(hidden_feat * mask_feat_dim[best_k], net.W)

        outputs = net.linear(hidden_feat_rec)
        
        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0].item(), inputs.size()[0])
        top5.update(acc5[0].item(), inputs.size()[0])

        msg = 'Top1: {:.3f}% | Top5: {:.3f}%'.format(top1.avg, top5.avg)
        utils.progress_bar(batchIdx, len(testloader), msg)
        
    return top1.avg, top5.avg

### --------------------------------------------------------------------------------------------

        
#-----------------------------------------------------------------------------------------------                       
#-----------------------------------------------------------------------------------------------            
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------                         
#-----------------------------------------------------------------------------------------------          
          
def main(gpu, arch, dataset, test_dir, batchsize, best_k, resumePth=None): 

    best_acc = 0  # best test accuracy
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    testloader, nb_cls = get_dataloader(dataset, test_dir, batchsize)
    
    # define net
    if arch == "PreActResNet18":
        net = resnet.PreActResNet18(h_dim=num_pc, num_classes=nb_cls)
        net.cuda()
    
    h_dim = net.h_dim

    # generate mask 
    mask_feat_dim = []
    for i in range(h_dim): 
        tmp = torch.cuda.FloatTensor(1, h_dim).fill_(0)
        tmp[:, : (i + 1)] = 1
        mask_feat_dim.append(tmp)
    
    # load model
    if resumePth: 
        param = torch.load(resumePth)
        net.load_state_dict(param['net'])
        print ('\t---Loading network weight from {}'.format(resumePth))
        
    with torch.no_grad(): 
        mean_feat = Statistics(testloader, net)
        net.mean_feat = mean_feat
        top1_acc, top5_acc = Test(best_k, net, testloader, mask_feat_dim)

    msg = '\t--- Test set: Top1: {:.3f}% | Top5: {:.3f}%'.format(top1_acc, top5_acc)
    print(msg)
    
    return top1_acc
    
    
    
if __name__ == '__main__': 
                        
    parser = argparse.ArgumentParser(description='PyTorch Classification on Test Set')

    # data
    parser.add_argument('--test-dir', type=str, default='../data/CIFAR10/test', help='val directory')  
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'Clothing1M'], default='CIFAR100', help='which dataset?')
    parser.add_argument('--batchsize', type=int, default=320, help='batch size')

    # model
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50', 'PreActResNet18'], default='PreActResNet18', help='which archtecture?')
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')
    parser.add_argument('--KList', type=int, default=None, nargs='+', help='best k of each model')
    parser.add_argument('--resumePthList', type=str, nargs='+', help='resume path (list) of different models (running)')
    
    args = parser.parse_args()
    print (args)

    acc_list = []
    for i in range(len(args.resumePthList)): 
        
        pth = args.resumePthList[i]
        
        print ('\nEvaluation of {}'.format(pth))

        if args.KList is not None:
            k = args.KList[i]
            print ('\nBest K is {:d}'.format(k))
        else:
            k = None

        acc = main(gpu = args.gpu, 
        
                   arch = args.arch, 
                     
                   dataset = args.dataset, 
                     
                   test_dir = args.test_dir, 
                     
                   batchsize = args.batchsize,

                   best_k = k,
                     
                   resumePth = os.path.join(pth, 'netBest.pth'))
                     
        acc_list.append(acc)
                
    
    print ('Final Perf: ')
    print ('\t --- Acc Avg is {:.3f}, Acc Std is {:.3f}....'.format(np.mean(acc_list), np.var(acc_list) ** 0.5))

    if args.KList is not None:
        print ('\t --- K Avg is {:.3f}, K Std is {:.3f}....'.format(np.mean(args.KList) + 1, np.var(args.KList) ** 0.5)) # need to add 1 since, nb of channels = index of channels + 1
