from torch import nn
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

#注意力机制
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModuleClassific(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out
#服务于CBAM模块
class ChannelAttentionModuleClassific(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModuleClassific, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

#多元数据融合
class Deeplab_Torch_Multisource(nn.Module):
    def __init__(self,n_class=13, multisource = True):
        if multisource == True:
            channel_num = 34
        else:
            channel_num = 21
        super(Deeplab_Torch_Multisource, self).__init__()
        self.n_class = n_class
        self.conv7_3 = nn.Conv2d(7, 3, kernel_size=1, stride=1)
        self.img_HR = deeplabv3_resnet50(pretrained=True, num_classes=21)
        self.img_LS = deeplabv3_resnet50(pretrained=True, num_classes=21)


        self.conv1 = nn.Conv2d(21, 13, kernel_size=3, stride=2,padding =1)
        self.conv_block_fc = nn.Sequential(
            nn.Conv2d(channel_num, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.cbam1 = CBAM(34)
    def forward(self, x_ls, lbr=None):
        x = self.img_LS(x_ls)['out']
        outputs = []
        if lbr is not None:
            lbr = self.conv7_3(lbr)
            x_GE = self.img_HR(lbr)['out']
            x_GE = self.conv1(x_GE)
            x = torch.cat([x, x_GE],1)
            x = self.cbam1(x)
        x = self.conv_block_fc(x)
        outputs.append(x)
        return tuple(outputs)