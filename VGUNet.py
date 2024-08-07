import math
import numpy as np
import torch
#import umap
#import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

#----------------------------------------------------------------------
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        # super(BasicConv2d, self).__init__()
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.prelu_weight = nn.Parameter(torch.Tensor(1).fill_(0.25))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #return F.relu(x, inplace=True)
        return F.prelu(x, self.prelu_weight)


class Regress(nn.Module):
    """Auxillary regression branch for orientation and distance"""

    def __init__(self, in_channels):
        super().__init__()

        self.conv = BasicConv2d(in_channels, 128)  # in_channels=128
        #self.conv = DoubleConv(in_channels, 128)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 2)
        self.prelu_weight = nn.Parameter(torch.Tensor(1).fill_(0.25))

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        #x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = F.adaptive_avg_pool2d(x, (4, 4)) # 256 x 4 x4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        # x = F.relu(self.fc1(x), inplace=True)
        x = F.prelu(self.fc1(x), self.prelu_weight)
        # N x 1024
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        x = self.fc3(x)
        # N x 1000 (num_classes)

        return x

#------------------------------------------------------------------------------------


class SpatialGCN(nn.Module):
    def __init__(self, plane,inter_plane=None,out_plane=None):
        super(SpatialGCN, self).__init__()
        if inter_plane==None:
            inter_plane = plane #// 2
        if out_plane==None:
            out_plane = plane
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.conv_wgl = nn.Linear(inter_plane,out_plane)
        self.bn1 = nn.BatchNorm1d(out_plane)
        self.conv_wgl2 = nn.Linear(out_plane, out_plane)
        self.bn2 = nn.BatchNorm1d(out_plane)
        self.softmax = nn.Softmax(dim=2)

        self.prelu_weight = nn.Parameter(torch.Tensor(1).fill_(0.25))


    def forward(self, x):
        node_k = self.node_k(x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)   #####nosym better, softmax better,only one gcn better
        node_q = self.node_q(x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)#
        # print("input:",x.shape,node_k.shape)
        node_v = self.node_v(x)  # x#
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)  ##b N C
        node_q = node_q.view(b, c, -1)  ###b c N
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)  ##b N C
        Adj = torch.bmm(node_k, node_q)  ###Q*K^T

        # test using cosine=(a*b)/||a||*||b|| to construct adjacency
        # Adj = torch.bmm(node_k,node_q)#ab_ij=node_i*node_j
        # batch_row_norm = torch.norm(node_k,dim=-1).unsqueeze(-1)
        # Adj = torch.div(Adj,torch.bmm(batch_row_norm,batch_row_norm.permute(0,2,1)))

        Adj = self.softmax(Adj)  ###adjacency matrix of size b N N

        # max = torch.max(Adj, dim=2)
        # min = torch.min(Adj, dim=2)
        # Adj = (Adj - min.values[:, :, None]) / max.values[:, :, None]  # normalized adjacency matrix
        # Adj[Adj<0.5]=0

        AV = torch.bmm(Adj,node_v)###AX
        #AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2))###AXW b n C
        AVW= F.prelu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2), self.prelu_weight)
        AVW = F.dropout(AVW)
        # add one more layer
        AV = torch.bmm(Adj,AVW)
        #AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW= F.prelu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2), self.prelu_weight)
        AVW = F.dropout(AVW)
        # end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        AVW = AVW.view(b, c, h, -1)
        return AVW

class VGUNet(nn.Module):
    # def __init__(self, in_ch=2, out_ch=2,base_nc=64,fix_grad=True)
    def __init__(self, in_ch, out_ch, base_nc=64, bilinear=True, fix_grad=True):
        super(VGUNet, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bilinear = bilinear


        self.fix_grad = fix_grad
        self.conv1 = DoubleConv(in_ch, base_nc)
        # self.conv0 = nn.Conv2d(in_ch, base_nc, 1, stride=1, padding=0, bias=False) # 3 (320,320) -> 64(320,320)
        # self.pool0 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling 64 (160,160)
        # self.conv1 = nn.Conv2d(base_nc, base_nc, 1, stride=1, padding=0, bias=False)# 64(160,160)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False) 
        self.conv2 = DoubleConv(base_nc, 2 * base_nc)
        self.pool2 = nn.Conv2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv3 = DoubleConv(2 * base_nc, 4 * base_nc)
        self.pool3 = nn.Conv2d(4 * base_nc, 4 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        # if self.fix_grad==True:
        #     for p in self.parameters():
        #         p.requires_grad=False
        self.sgcn3 = SpatialGCN(2 * base_nc)
        self.sgcn2 = SpatialGCN(4 * base_nc)
        self.sgcn1 = SpatialGCN(4 * base_nc)  ###changed with spatialGCN

        # factor = 2 if bilinear else 1
        self.regress = Regress(4* base_nc)


        self.up6 = nn.ConvTranspose2d(4 * base_nc, 4 * base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv6 = DoubleConv(8 * base_nc, 4 * base_nc)
        self.up7 = nn.ConvTranspose2d(4 * base_nc, 2 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv7 = DoubleConv(4 * base_nc, 2 * base_nc)
        self.up8 = nn.ConvTranspose2d(2 * base_nc, base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv8 = DoubleConv(2 * base_nc, base_nc)
        # self.up9 = nn.ConvTranspose2d(base_nc, base_nc, 2, stride=2,padding=0)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1, padding=0)

        

    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        # c0=self.conv0(x)  ## 2 nc
        # p0=self.pool0(c0)
        # c1=self.conv1(p0)
        #print("c1 shape:", c1.shape)
        p1=self.pool1(c1)  ##
        # print("p1 shape:", p1.shape)
        c2=self.conv2(p1) ##nc 2nc
        # print("c2 shape:", c2.shape)
        p2=self.pool2(c2)
        # print("p2 shape:", p2.shape)
        c3=self.conv3(p2) ##2nc 2nc
        # print("c3 shape:", c3.shape)
        p3=self.pool3(c3)
        #print("p3 shape:", p3.shape)
        c4=self.sgcn1(p3)   ###spatial gcn 4nc
        #print("c4 shape:", c4.shape)
        x_regress=self.regress(c4)
        up_6= self.up6(c4)
        # print("up_6 shape:", up_6.shape)
        # print("Every thing in VGUnet ok untile now")
        # print("sgcn2(c3) shape:",self.sgcn2(c3).shape)
        merge6= torch.cat([up_6, self.sgcn2(c3)], dim=1) ##gcn
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, self.sgcn3(c2)], dim=1)
        c7=self.conv7(merge7)
        #x_regress=self.regress(c7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)
        # c9= self.up9(c8)
        # c10=self.conv9(c9)

        return c9, x_regress
        # return c10, x_regress

