import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
import os
import sys
sys.path.append(os.getcwd())
from siamrpn.config import config


BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class FuseLayer(nn.Module):
    def __init__(self,channel1,channel2,scale=2):
        super(FuseLayer,self).__init__()

        self.scale = scale

        self.downsample = []

        while scale>1:
            scale = scale/2
            self.downsample.append(nn.Sequential(
                conv3x3(int(self.scale/scale/2)*channel1,int(self.scale/scale)*channel1,stride=2)
            ))

        self.downsample.append(nn.Sequential(nn.BatchNorm2d(self.scale*channel1)))

        self.upsample = nn.Sequential(
            nn.Conv2d(channel2,int(channel2/self.scale),kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(int(channel2/self.scale)),
            nn.UpsamplingNearest2d(scale_factor=self.scale)
        )

    def forward(self,x,y):
        for m in self.downsample:
            m.cuda()
            x = m(x)
        downx = x
        upy = self.upsample(y)
        return downx,upy

class HRNetV(nn.Module):
    def __init__(self,device="cpu"):
        super(HRNetV,self).__init__()
        self.downsample = nn.Sequential(
            conv3x3(3,64,2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64,64,2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ).to(device)

        self.transition1_1 = nn.Sequential(
            conv3x3(256,32,1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        ).to(device)

        self.transition1_2 = nn.Sequential(
            conv3x3(256,64,2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ).to(device)

        self.transition2_1 = nn.Sequential(
            nn.ReLU(True)
        ).to(device)

        self.transition2_2 = nn.Sequential(
            nn.ReLU(True),
            conv3x3(64,128,2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        ).to(device)
        
        self.transition3_2 = nn.Sequential(
            conv3x3(128,256,2),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        ).to(device)

        downsample = nn.Sequential(
            nn.Conv2d(64,256,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(256,momentum=BN_MOMENTUM)
        ).to(device)

        self.layer1 = nn.Sequential(
            Bottleneck(64,64,downsample=downsample),
            Bottleneck(256,64),
            Bottleneck(256,64),
            Bottleneck(256,64)
        ).to(device)

        self.stage2_y_b = nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64),
            BasicBlock(64,64),
            BasicBlock(64,64)
        ).to(device)

        self.stage2_x_b = nn.Sequential(
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32,32)
        ).to(device)

        self.stage3_x_b = []
        self.stage3_y_b = []
        self.fuse_stage3 = []
        self.relu_stage3_x = []
        self.relu_stage3_y = []
        for i in range(4):
            self.stage3_x_b.append(nn.Sequential(
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32,32)).to(device)
            )
            self.stage3_y_b.append(nn.Sequential(
            BasicBlock(128,128),
            BasicBlock(128,128),
            BasicBlock(128,128),
            BasicBlock(128,128)).to(device))
            self.fuse_stage3.append(
                FuseLayer(32,128,4).to(device)
            )
            self.relu_stage3_x.append(nn.ReLU(True).to(device))
            self.relu_stage3_y.append(nn.ReLU(True).to(device))

        self.fuse_stage2 = FuseLayer(32,64).to(device)

        self.stage4_x_b = []
        self.stage4_y_b = []
        self.fuse_stage4 = []
        self.relu_stage4_x = []
        self.relu_stage4_y = []
        for i in range(3):
            self.stage4_x_b.append(nn.Sequential(
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32,32)).to(device)
            )
            self.stage4_y_b.append(nn.Sequential(
            BasicBlock(256,256),
            BasicBlock(256,256),
            BasicBlock(256,256),
            BasicBlock(256,256)).to(device))
            self.fuse_stage4.append(
                FuseLayer(32,256,8).to(device)
            )
            self.relu_stage4_x.append(nn.ReLU(True).to(device))
            self.relu_stage4_y.append(nn.ReLU(True).to(device))

    def forward(self,input):
        #print(input.shape)
        input = self.downsample(input)
        input = self.layer1(input)

        """Transition1"""
        x = self.transition1_1(input)
        y = self.transition1_2(input)

        """Stage2"""
        x = self.stage2_x_b(x)
        y = self.stage2_y_b(y)
        downx,upy = self.fuse_stage2(x,y)
        x = x + upy
        y = y + downx

        """Transition2"""
        x = self.transition2_1(x)
        y = self.transition2_2(y)


        """Stage3"""
        for i in range(4):
            x = self.stage3_x_b[i](x)
            y = self.stage3_y_b[i](y)
            downx,upy = self.fuse_stage3[i](x,y)
            x = x + upy
            y = y + downx
            x = self.relu_stage3_x[i](x)
            y = self.relu_stage3_y[i](y)

        """Transition3"""
        y = self.transition3_2(y)

        """Stage4"""
        for i in range(3):
            x = self.stage4_x_b[i](x)
            y = self.stage4_y_b[i](y)
            downx,upy = self.fuse_stage4[i](x,y)
            x = x + upy
            y = y + downx
            x = self.relu_stage4_x[i](x)
            y = self.relu_stage4_y[i](y)

        #print(x.shape)
        #print(y.shape)

        return x

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        
    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)  #x [32,32,62,62]    kernel [32,32,12,12]
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out    

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel) 
        search = self.conv_search(search) 
        feature = self.xcorr_depthwise(search, kernel) 
        out = self.head(feature)
        return out

class HRSiamRPNNet(nn.Module):
    def __init__(self, init_weight=False,device="cpu"):
        super(HRSiamRPNNet, self).__init__()
        self.featureExtract = HRNetV(device=device)
        self.crop = nn.Sequential(T.CenterCrop(14)).to(device)
        self.anchor_num = config.anchor_num    #每一个位置有5个anchor
        """ 模板的分类和回归"""
        self.examplar_cla = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0).to(device)
        self.examplar_reg = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0).to(device)
        """ 搜索图像的分类和回归"""
        self.instance_cla = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0).to(device)
        self.instance_reg = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0).to(device)

        self.dwcorr_cla = DepthwiseXCorr(32, 32, 2*self.anchor_num).to(device)
        self.dwcorr_reg = DepthwiseXCorr(32, 32, 4*self.anchor_num).to(device)
        if init_weight:
            self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1) #xavier是参数初始化，它的初始化思想是保持输入和输出方差一致，这样就避免了所有输出值都趋向于0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)     #偏置初始化为0
            elif isinstance(m, nn.BatchNorm2d):      #在激活函数之前，希望输出值由较好的分布，以便于计算梯度和更新参数，这时用到BatchNorm2d函数
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    """——————————前向传播用于训练——————————————————"""
    def forward(self, template, detection):
        N = template.size(0)    # batch=32
        template_feature = self.featureExtract(template)    #[32,32,32,32]
        template_feature = self.crop(template_feature)      #[32,32,14,14]
        detection_feature = self.featureExtract(detection)  #[32,32,64,64]
        """对应模板和搜索图像的分类"""
        # [32,2k,51,51]
        pred_score = self.dwcorr_cla(template_feature,detection_feature)
        """对应模板和搜索图像的回归----------"""
        #[32,4k,51,51]
        pred_regression = self.dwcorr_reg(template_feature,detection_feature)

        return pred_score, pred_regression
    """—————————————初始化————————————————————"""
    def track_init(self, template):
        N = template.size(0) #1
        template_feature = self.featureExtract(template)# [1,32, 32, 32]
        self.template_feature = self.crop(template_feature) 
        
    """—————————————————跟踪—————————————————————"""
    def track_update(self, detection):
        N = detection.size(0)
        # [1,32,64,64]
        detection_feature = self.featureExtract(detection)
        """---------与模板互相关----------"""
        # input=[1,256,22,22] filter=[2*5,256,4,4] gropu=1 得output=[1,2*5,19,19]
        pred_score = self.dwcorr_cla(self.template_feature,detection_feature)
        # input=[1,256,22,22] filter=[4*5,256,4,4] gropu=1 得output=[1,4*5,19,19]
        pred_regression = self.dwcorr_reg(self.template_feature,detection_feature)
        #score.shape=[1,10,19,19],regression.shape=[1,20,19,19]
        return pred_score, pred_regression


if __name__ == '__main__':

    model = HRSiamRPNNet()
    z_train = torch.randn([32,3,127,127])  #batch=8
    x_train = torch.randn([32,3,255,255])
    # 返回shape为[32,32,32,32] [32,32,64,64]  
    pred_score_train, pred_regression_train = model(z_train,x_train)
    print(pred_score_train.shape)
    print(pred_regression_train.shape)
    print("==========================")
    z_test = torch.randn([1,3,127,127])
    x_test = torch.randn([1,3,255,255])
    model.track_init(z_test)
    pred_score_test, pred_regression_test = model.track_update(x_test)
    print(pred_score_test.shape)
    print(pred_regression_test.shape)



