# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:18:52 2020

@author: 30262
"""







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



import scipy.io as io



test1 = io.loadmat('./res.mat')['testdata']


magdata = test1[0]

test2 = io.loadmat('./true.mat')['truemodel']*-1
magdata2 = test2[0]


SR_probs = torch.from_numpy(test1).float()
SR_probs = SR_probs.permute(0,3,2,1)

test2 = torch.from_numpy(test2).float()
test2 = test2.permute(0,3,2,1)

GT_flat = SR_probs.view(SR_probs.size(0),-1)
b_y1 = test2.view(test2.size(0),-1)

criterion = torch.nn.BCELoss()
loss = criterion(GT_flat, b_y1)


acc = get_accuracy(SR_probs,test2)

import matplotlib.pyplot as plt


#[i for i,v in enumerate(valacc)) if v < 0.99]



s=magdata
s2=magdata2*-1



#mlab.show()

mlab.figure(size=[900,700],bgcolor=(1.0,1.0,1.0),fgcolor=(0,0,0))

mlab.colorbar(title='Probability')
src = mlab.pipeline.scalar_field(s)
#mlab.pipeline.iso_surface(src, opacity=0.1,colormap = 'coolwarm',vmin=0,vmax=1)
mlab.pipeline.iso_surface(src,colormap = 'coolwarm', opacity=0,line_width=0,vmin=0,vmax=1)

src = mlab.pipeline.scalar_field(s2)
mlab.pipeline.iso_surface(src, contours=[s2.min()+0.1*s2.ptp()],color=(0,0,0), opacity=0.5,line_width=0)

src = mlab.pipeline.scalar_field(s)
#mlab.pipeline.iso_surface(src, opacity=0.1,colormap = 'coolwarm',vmin=0,vmax=1)
mlab.pipeline.iso_surface(src,colormap = 'coolwarm', opacity=0,line_width=0,vmin=0,vmax=1)



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
mlab.savefig(filename='fieldres.bmp')
mlab.show()

c=1
# mlab.savefig(filename='ball1.bmp')
#cam,foc = mlab.move()
