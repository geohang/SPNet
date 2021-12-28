import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import os
from evaluation import *
from mayavi import mlab
from mayavi.tools import pipeline
import pandas as pd
import numpy as np


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block1,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv1(nn.Module):
    def __init__(self,ch_in,ch_out,numsize1,numsize2):
        super(up_conv1,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=(numsize1,numsize2)),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=31,output_ch=31):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)


        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)

       # self.Up5 = up_conv(ch_in=1024,ch_out=512)
       #  self.Up_conv5 = conv_block(ch_in=512, ch_out=512)
       #
       #  self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)

        self.Up3 = up_conv1(ch_in=256,ch_out=128,numsize1=4,numsize2=3)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv1(ch_in=128,ch_out=64,numsize1=9,numsize2=7)
        #self.Up2 = up_conv1(ch_in=128,ch_out=64,numsize=17)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)#8
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)#4
        x3 = self.Conv3(x3)

        d4 = self.Up_conv4(x3)
        # x4 = self.Maxpool(x3)#2
        # x4 = self.Conv4(x4)

        # d5 = self.Up_conv5(x4)
        #
        # d4 = self.Up4(d5)#4
        # d4 = torch.cat((x3,d4),dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)#8
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)#17
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
#        d1  = d1 .view(d1.size(0), -1) # 将（batch，32,7,7）展平为（batch，32*7*7）
#        output = self.out(d1)

        return d1

unet = U_Net().cuda()



unet.load_state_dict(torch.load('params1.pkl'))

X_test = torch.load('Xtest')
Y_test = torch.load('Ytest')

Y_pred = torch.zeros(Y_test.shape)

test_loader = DataLoader(TensorDataset(X_test, Y_test), 1, shuffle = False)

criterion = torch.nn.BCELoss()
unet.eval()
val_loss = []
valacc = []
#        correct = 0
with torch.no_grad():
    for i, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = unet(b_x)
        SR_probs = F.sigmoid(output)
        Y_pred[i] = SR_probs
        GT_flat = SR_probs.view(SR_probs.size(0),-1)
        b_y1 = b_y.view(b_y.size(0),-1)

        loss = criterion(GT_flat, b_y1)
        val_loss.append(loss.item())

        valacc.append(get_accuracy(SR_probs,b_y))

import matplotlib.pyplot as plt

plt.hist(valacc)
plt.show()
plt.hist(val_loss)
plt.plot(valacc)
val_loss = np.array(val_loss)
valacc = np.array(valacc)
np.savetxt('testresults.txt',np.vstack((np.array(valacc),np.array(val_loss))).T)
#[i for i,v in enumerate(valacc)) if v < 0.99]

Y_pred = Y_pred.permute(0,3,2,1)
Y_test = Y_test.permute(0,3,2,1)
X_test = X_test.permute(0,3,2,1)


##### following is the 3D plot codes for model
num = 50
print(valacc[num])
magdata = Y_pred[num].cpu().detach().numpy()
magdata2 = Y_test[num].cpu().detach().numpy()

s=magdata
s2=magdata2



#mlab.show()

mlab.figure(size=[900,700],bgcolor=(1.0,1.0,1.0),fgcolor=(0,0,0))

src = mlab.pipeline.scalar_field(s)
mlab.pipeline.iso_surface(src,colormap = 'coolwarm', opacity=0,line_width=0)
mlab.colorbar(title='Probability')

src = mlab.pipeline.scalar_field(s2)
mlab.pipeline.iso_surface(src, contours=[s2.min()+0.1*s2.ptp()],color=(0,0,0), opacity=0.5,line_width=0)


#mlab.pipeline.iso_surface(src, opacity=0.1,colormap = 'coolwarm',vmin=0,vmax=1)


src = mlab.pipeline.scalar_field(s)
mlab.pipeline.image_plane_widget(src,
                            plane_orientation='x_axes',
                            slice_index=4,
                            colormap = 'coolwarm',
                            transparent=False,vmin=0,vmax=1
                        )



mlab.pipeline.image_plane_widget(src,
                            plane_orientation='y_axes',
                            slice_index=5,
                            colormap = 'coolwarm',
                            transparent=False,vmin=0,vmax=1
                        )




mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=8,
                            colormap = 'coolwarm',
                            transparent=False,vmin=0,vmax=1
                        )


mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=24,
                            colormap = 'coolwarm',
                            transparent=False,vmin=0,vmax=1
                        )





mlab.axes(ranges=[0,30,0,24,0,80],xlabel='X (cm)',ylabel='Y (cm)',zlabel='Z (cm)',color=(0,0,0),line_width=4)

mlab.view(azimuth=16.888835156942815, elevation= 104.85988517332146, distance= 70.86243999999321, focalpoint=(3,5,16),roll=-4.591915633303024)

mlab.show()

##### following is the 3D plot codes for normized SP response

magdata3 = X_test[num].cpu().detach().numpy()

s3 = magdata3
src = mlab.pipeline.scalar_field(s3)
#mlab.pipeline.iso_surface(src, opacity=0.1,colormap = 'coolwarm',vmin=0,vmax=1)
mlab.pipeline.iso_surface(src,colormap = 'coolwarm', opacity=0,line_width=0)
mlab.colorbar(title='SP Signal')


mlab.pipeline.image_plane_widget(src,
                            plane_orientation='x_axes',
                            slice_index=3,
                            colormap = 'coolwarm',
                            transparent=False
                        )

mlab.pipeline.image_plane_widget(src,
                            plane_orientation='y_axes',
                            slice_index=5,
                            colormap = 'coolwarm',
                            transparent=False
                        )




mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=8,
                            colormap = 'coolwarm',
                            transparent=False
                        )


mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=24,
                            colormap = 'coolwarm',
                            transparent=False
                        )


# mlab.xlabel('x')
# mlab.outline()
mlab.zlabel('Z (cm)')
mlab.xlabel('X (cm)')
mlab.ylabel('Y (cm)')
mlab.axes(ranges=[0,30,0,24,0,80])
mlab.show()

