# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 10:48:51 2020

@author: hang chen
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import os
from evaluation import *
#from mayavi import mlab
#from mayavi.tools import pipeline
log_dir=os.path.join('logs','train')
trainwriter= SummaryWriter(log_dir)
log_dir=os.path.join('logs','val')
valwriter= SummaryWriter(log_dir)





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

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d4 = self.Up_conv4(x3)


        d3 = self.Up3(d4)#8
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)#17
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        return d1
   
import scipy.io as io

## load the training datasets
all = io.loadmat('./SPtest.mat')

ydata = all['model']*-1
xdata = all['res']

torch.manual_seed(1)    # reproducible


EPOCH = 100
BATCH_SIZE = 1000
LR = 0.0001

x = torch.from_numpy(xdata).float()
y = torch.from_numpy(ydata).float()
xtemp = x.permute(0,3,2,1)
ytemp = y.permute(0,3,2,1)

ntrain = 90000
nval = 5000
ntest = 5000

index = [i for i in range(len(x))]

random.shuffle(index)
np.save('index',index)

X_train = xtemp[index[:ntrain]]
Y_train = ytemp[index[:ntrain]]

X_val = xtemp[index[ntrain:ntrain+nval]]
Y_val = ytemp[index[ntrain:ntrain+nval]]

X_test = xtemp[index[ntrain+nval:ntrain+nval+ntest]]
Y_test = ytemp[index[ntrain+nval:ntrain+nval+ntest]]

unet = U_Net()
unet = unet.cuda()
    

print(unet)

init_weights(unet, 'normal', 0.02)

params = list(unet.parameters())
print(len(params))
print(params[0].size())
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


    
train_loader = DataLoader(TensorDataset(X_train,Y_train), BATCH_SIZE, shuffle = True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), BATCH_SIZE, shuffle = True)

criterion = torch.nn.BCELoss()

train_loss = 0
valid_loss = 0

losses = []
val_losses = []

for epoch in range(EPOCH):
    unet.train()
    train_loss = 0.
    acc = 0.

    for i, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = unet(b_x)   
        SR_probs = F.sigmoid(output)
        GT_flat = SR_probs.view(SR_probs.size(0),-1)
        b_y1 = b_y.view(b_y.size(0),-1)
        loss = criterion(GT_flat, b_y1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        train_loss += loss.data 
        acc += get_accuracy(SR_probs,b_y)

        if (i+1) % 10 == 0:

            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch + 1, EPOCH, 
                                                               i + 1, len(X_train)//BATCH_SIZE, 
                                                               loss.data))
    
    train_loss = train_loss/(i+1)
    acc = acc/(i+1)
    trainwriter.add_scalar("Loss",train_loss,epoch)
    trainwriter.add_scalar("Accuracy",acc,epoch)
    
    print('Train set: Average loss: {:.4f}\n'.format(train_loss))

    # evaluate accuracy in validataion datasets every epoch
    unet.eval()
    valid_loss = 0
    valacc = 0
    with torch.no_grad():
        for i, (b_x, b_y) in enumerate(val_loader):          
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = unet(b_x)   
            SR_probs = F.sigmoid(output)
            GT_flat = SR_probs.view(SR_probs.size(0),-1)
            b_y1 = b_y.view(b_y.size(0),-1)

            loss = criterion(GT_flat, b_y1)
            val_losses.append(loss.item())
            valid_loss += loss.data 
            valacc += get_accuracy(SR_probs,b_y)

    
    valid_loss = valid_loss/(i+1)
    valacc = valacc/(i+1)
    valwriter.add_scalar("Loss",valid_loss,epoch)
    valwriter.add_scalar("Accuracy",valacc,epoch)
    print('Test set: Average loss: {:.4f}\n'.format(valid_loss))
    print('Accurcy:',acc,valacc)
    scheduler.step()      
        
 


torch.save(X_test , 'Xtest')
torch.save(Y_test, 'Ytest')
torch.save(unet.state_dict(), 'params1.pkl')

valwriter.close()
trainwriter.close() 


# load the lab measured dataset
test = io.loadmat('./observed.mat')['observed']
test = (test - np.min(test))/(np.max(test)- np.min(test))
test = torch.from_numpy(test).float()
test = test.permute(0,3,2,1)
x_test = test.cuda()

outtest = unet(x_test)
SR_probs = F.sigmoid(outtest)
SR_probs = SR_probs.permute(0,3,2,1)
testres = SR_probs.cpu().detach().numpy()

# save the prediction from the lab measured dataset
test2 = io.loadmat('./test.mat')
test2['testdata'] = testres
io.savemat('res.mat',test2)

