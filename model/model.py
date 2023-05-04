import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import math
import model.cifar_resnet as cifar


class NetFeat(nn.Module):
    def __init__(self, arch, pretrained, dataset):
        super(NetFeat, self).__init__()
        if 'CIFAR' in dataset:
            if 'resnet' in arch:
                if arch == 'resnet18':
                    net = cifar.resnet18()

                resnet_feature_layers = ['conv1','conv2_x','conv3_x','conv4_x','conv5_x']
                resnet_module_list = [getattr(net,l) for l in resnet_feature_layers]
                last_layer_idx = resnet_feature_layers.index('conv5_x')
                featExtractor = nn.Sequential(*(resnet_module_list[:last_layer_idx+1] + [nn.AdaptiveAvgPool2d((1, 1))]))

                self.feat_net = featExtractor
                self.feat_dim = 512

        elif dataset == 'Clothing1M':
            if arch == 'resnet50':
                net = imagenet.resnet50(pretrained=pretrained)
                self.feat_dim = 2048

            elif arch == 'resnet18':
                net = imagenet.resnet18(pretrained=pretrained)
                self.feat_dim = 512

            resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
            resnet_module_list = [getattr(net,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index('layer4')
            featExtractor = nn.Sequential(*(resnet_module_list[:last_layer_idx+1] + [nn.AvgPool2d(7, stride=1)]))

            self.feat_net = featExtractor

    def train(self, mode=True, freeze_bn=False):
        """
        Override the default train() to freeze the BN parameters
        """
        super(NetFeat, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x):
        x = self.feat_net(x)
        x = torch.flatten(x, 1)
        return x

# class Encoder(nn.Module):
#     def __init__(self, ):
#         super(Encoder, self).__init__()
#         self.args = args
#         self.main = nn.Sequential(
#         ) # encoder structure

#     def forward(self, x):
#         return self.main(x)

# class Decoder(nn.Module):
#     def __init__(self, ):
#         super(Decoder, self).__init__()
#         self.args = args
#         self.main = nn.Sequential(
#         ) # decoder structure

#     def forward(self, x):
#         return self.main(x)

class LPCA(nn.Module):
    def __init__(self, feat_dim, num_pc, mask_feat_dim, dist=None, dropout=0.0):
        super(KPCA, self).__init__()
        self.feat_dim = feat_dim
        self.num_pc = num_pc
        # Initialize manifold parameter
        self.W = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.feat_dim, self.num_pc))) #W matrix, 512 x l (number principal components)

        # self.encoder = Encoder()
        # self.decoder = Decoder()

    def forward(self, x_feat):
        x_feat = x_feat - torch.mean(x_feat, dim=0)
        # covariance matrix
        Cov = torch.mm(x_feat.t(), x_feat)

        hidden_feat = torch.mm(x_feat, self.W)

        if dist is not None:
            k = np.random.choice(range(len(mask_feat_dim)), p=dist)
            mask_k = mask_feat_dim[k]
            hidden_feat_masked = hidden_feat * mask_k
        else:
            hidden_feat_masked = F.dropout(hidden_feat, p=dropout, training=True)

        x_rec = torch.mm(hidden_feat_masked, self.W.t())

        lpca_loss = torch.trace(Cov - torch.mm(torch.mm(self.W, self.W.t()), Cov))/x_rec.size(0)  # KPCA

        return x_rec, lpca_loss


class NetClassifier(nn.Module):
    def __init__(self, feat_dim, nb_cls):
        super(NetClassifier, self).__init__()
        self.weight = torch.nn.Parameter(nn.Linear(feat_dim, nb_cls, bias=False).weight.T, requires_grad=True) # dimension feat_dim * nb_cls

    def getWeight(self):
        return self.weight, self.bias, self.scale_cls

    def forward(self, feature):
        batchSize, nFeat = feature.size()
        clsScore = torch.mm(feature, self.weight)

        return clsScore

# # Accumulate trainable parameters in 2 groups. 1. Manifold_params 2. Network param
# def param_state(model):
#     param_g, param_e1 = [], []
#     for name, param in model.named_parameters():
#         if param.requires_grad and name != 'W':
#             param_e1.append(param)
#         elif name == 'W':
#             param_g.append(param)
#     return param_g, param_e1

# def stiefel_opti(stief_param, lrg=1e-4):
#     dict_g = {'params': stief_param, 'lr': lrg, 'momentum': 0.9, 'weight_decay': 0.0005, 'stiefel': True}
#     return stiefel_optimizer.AdamG([dict_g])  # CayleyAdam


if __name__ == '__main__':

    data = torch.randn(3, 3, 32, 32).cuda()
    net_feat = NetFeat(arch='resnet18', pretrained=False, dataset='CIFAR100')
    net_lpca = LPCA(net_feat.feat_dim, num_pc=50)
    net_cls = NetClassifier(net_feat.feat_dim, 100)

    net_feat.cuda()
    net_lpca.cuda()
    net_cls.cuda()

    feat = net_feat(data)
    print (feat.size())
    x_rec, lpca_loss = net_lpca(feat)
    preds = net_cls(x_rec)
    print (preds.size())
