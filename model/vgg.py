import torch
import torch.nn as nn
#import torch.nn.functional as F

import torchvision.models as models
#import numpy as np
#import torch.nn.init as init
import math

class VGG(nn.Module):
    def __init__(self, h_dim, arch, pretrained, dataset, vgg_dropout=0):
        super(VGG, self).__init__()
        self.vgg_dropout = vgg_dropout

        net = models.vgg19_bn(pretrained=pretrained)

        classifier = []
        for m in net.classifier.children() :
            if 'dropout' in m.__module__ :
                if self.vgg_dropout > 0 :
                    m = nn.Dropout(self.vgg_dropout)
            classifier.append(m)
        self.forward1 = nn.Sequential(*classifier[:2])
        self.forward2 = nn.Sequential(*classifier[3:5])

        if self.vgg_dropout > 0 :
            self.dropout1 = classifier[2]
            self.dropout2 = classifier[5]

        self.feat_net = net
        self.feat_dim = 4096

        self.h_dim = h_dim
        self.W = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.h_dim, self.feat_dim)))
        self.fc = nn.Linear(self.feat_dim, nb_cls)

        #self.mask_feat_dim.append(tmp)

    def train(self, mode=True, freeze_bn=False) :
        """
        Override the default train() to freeze the BN parameters
        """
        super(Net, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn :
            for m in self.modules() :
                if isinstance(m, nn.BatchNorm2d) :
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


    def forward(self, x):
        x = self.feat_net.features(x)
        x = self.feat_net.avgpool(x)
        x = torch.flatten(x, 1)
        x_feat = self.forward1(x)
        if self.vgg_dropout > 0 :
            x_feat = self.dropout1(x_feat)
        x_feat = self.forward2(x_feat)
        if self.vgg_dropout > 0 :
            x_feat = self.dropout2(x_feat)

        outputs = self.fc(x_feat)

        return outputs, x_feat

        #outputs = out.view(out.size(0), -1)
    def forward_lpca(self, x, mean_feat = None) :
        x = self.feat_net.features(x)
        x = self.feat_net.avgpool(x)
        x = torch.flatten(x, 1)
        x_feat = self.forward1(x)
        if self.vgg_dropout > 0 :
            x_feat = self.dropout1(x_feat)
        x_feat = self.forward2(x_feat)

        if mean_feat is not None :
            x_feat = x_feat_raw - mean_feat
        else:
            x_feat = x_feat_raw - torch.mean(x_feat_raw, dim=0)  # feature centering

        Cov = torch.mm(x_feat.t(), x_feat) # Covariance matrix

        hidden_feat = torch.mm(x_feat, self.W.t())

        x_rec = torch.mm(hidden_feat, self.W)
        out = self.linear(x_rec)

        lpca_loss = torch.trace(Cov - torch.mm(torch.mm(self.W.t(), self.W), Cov))/out.size(0)  # learnable pca

        return out, lpca_loss


if __name__ == '__main__' :

    x = torch.randn(20, 3, 64, 64).cuda()

    # generate mask
    h_dim = 100

    # network
    net = Net(h_dim = h_dim,
              nb_cls = 10,
              pretrained = True)

    net.cuda()

    with torch.no_grad() :
        output, x_feat = net(x)
