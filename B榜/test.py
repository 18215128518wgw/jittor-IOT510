import glob
import random
import os
import numpy as np
import jittor as jt
import math
import itertools
import datetime
import sys
import cv2
import time
import jittor.transform as transform
import argparse

from jittor import init
from jittor import nn

from models import *
from datasets import *

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")
jt.flags.use_cuda = 1
parser=argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=280, help="epoch to start training from")
parser.add_argument("--input_path", type=str, default="./jittor_landscape_200k/val")
parser.add_argument("--output_path", type=str, default="./results")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()


def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

os.makedirs(f"{opt.output_path}/images_1000/", exist_ok=True)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()   
# criterion_GAN = nn.MSELoss()       --------------------这里更换了损失函数-------------------
criterion_pixelwise = nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
discriminator = Discriminator()

if opt.epoch != 0:
    # Load pretrained models
    generator.load(f"{opt.output_path}/flickr/saved_models/generator_{opt.epoch}.pkl")
    discriminator.load(f"{opt.output_path}/flickr/saved_models/discriminator_{opt.epoch}.pkl")


# Configure dataloaders
transforms = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

val_dataloader = ImageDataset(opt.input_path, mode="val", transforms=transforms).set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=1,
)

@jt.single_process_scope()
def eval(epoch):
    cnt = 1
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        fake_B = generator(real_A)
        
        # if i == 0:
            # img_sample = np.concatenate([real_A.data, fake_B.data], -2)
            # img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_sample.png", nrow=5)        

        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/images_1000/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1

eval(280)
