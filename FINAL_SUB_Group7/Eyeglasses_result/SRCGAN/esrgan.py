"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import Dataset
import pandas as pd
import time
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from classifier import *

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

torch.set_printoptions(linewidth=200)
os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--dataset_name", type=str,
                    default="MA", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=64,
                    help="high res. image height")
parser.add_argument("--hr_width", type=int, default=64,
                    help="high res. image width")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10,
                    help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=20,
                    help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23,
                    help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=10,
                    help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float,
                    default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float,
                    default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = 0.0000001
print(opt)
cost_fn = torch.nn.CrossEntropyLoss()
model = resnet18(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(
    "real_eyeglasses_classifier.pt"), strict=False)


def our_loss(img):
    loss = model(img)
    return loss


print(device)

hr_shape = (opt.hr_height, opt.hr_width)

generator = GeneratorRRDB(opt.channels, filters=64,
                          num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

feature_extractor.eval()

criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
print("1", torch.cuda.get_device_properties(0).total_memory)
print("2", torch.cuda.memory_reserved(0))
print("3", torch.cuda.memory_allocated(0))

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(
        "saved_models/generator_2.pth"))
    discriminator.load_state_dict(torch.load(
        "saved_models/discriminator_2.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def run():
    print("hell2")
    # freeze_support()
    torch.multiprocessing.freeze_support()
    print('loop2')


if __name__ == '__main__':
    run()
    dataloader = DataLoader(
        ImageDataset("../../data/%s" %
                     opt.dataset_name, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------
    #

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            # print(imgs_lr.shape)
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(
                np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(
                np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(
                pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)
            # print("shape_of_gen_hr", gen_hr.shape)
            # print("shape_of_img_hr", imgs_hr.shape)

            gen_hr_class_loss = our_loss(gen_hr)
            imgs_hr_class_loss = our_loss(imgs_hr)
            class_loss = 0
            if i != len(dataloader)-1:
                for j in range(opt.batch_size):
                    if(imgs_hr_class_loss[0][j][0] > imgs_hr_class_loss[0][j][1]):

                        class_loss = class_loss+torch.log(imgs_hr_class_loss[1][j][0]+epsilon)-torch.log(
                            gen_hr_class_loss[1][j][0]+epsilon)
                    else:
                        class_loss = class_loss+torch.log(imgs_hr_class_loss[1][j][1]+epsilon)-torch.log(
                            gen_hr_class_loss[1][j][1]+epsilon)
            loss_G = loss_content + opt.lambda_adv * loss_GAN + \
                opt.lambda_pixel * loss_pixel + class_loss/opt.batch_size
            # print("loss_G", loss_G, "loss_content", loss_content, "loss_GAN", loss_GAN,
            #       "loss_pixel", loss_pixel, "class_loss/opt.batch_size", class_loss/opt.batch_size)
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(
                pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(
                pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            if i == len(dataloader)-1:
                continue
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f,class_loss:%f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                    (class_loss/opt.batch_size).item()

                )
            )

            if batches_done % opt.sample_interval == 0:
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = denormalize(
                    torch.cat((imgs_lr, gen_hr, imgs_hr), -1))
                save_image(img_grid, "images/training/%d.png" %
                           batches_done, nrow=1, normalize=False)

            if batches_done % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(),
                           "saved_models/generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(),
                           "saved_models/discriminator_%d.pth" % epoch)
