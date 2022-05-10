from os import listdir
from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
from classifier import *
parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", type=str,
#                     default="../../data/Augmented_data/000002.jpg")
parser.add_argument("--checkpoint_model", type=str,
                    default="generator_19_eyeglasses.pth")
parser.add_argument("--channels", type=int, default=3,
                    help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23,
                    help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/output_gender", exist_ok=True)
os.makedirs("images/output_eye", exist_ok=True)
os.makedirs("images/output_eye_hat", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define model and load model checkpoint
generator1 = GeneratorRRDB(opt.channels, filters=64,
                           num_res_blocks=opt.residual_blocks).to(device)
generator1.load_state_dict(torch.load(
    "./saved_models/generator_19_esrgan.pth"))
generator1.eval()

generator2 = GeneratorRRDB(opt.channels, filters=64,
                           num_res_blocks=opt.residual_blocks).to(device)
generator2.load_state_dict(torch.load(
    "./saved_models/generator_19_gender.pth"))
generator2.eval()


generator3 = GeneratorRRDB(opt.channels, filters=64,
                           num_res_blocks=opt.residual_blocks).to(device)
generator3.load_state_dict(torch.load(
    "./saved_models/generator_19_eyeglasses.pth"))
generator3.eval()

generator4 = GeneratorRRDB(opt.channels, filters=64,
                           num_res_blocks=opt.residual_blocks).to(device)
generator4.load_state_dict(torch.load(
    "./saved_models/generator_20_hat_eye.pth"))
generator4.eval()


transform = transforms.Compose(
    [transforms.Resize(
        (16, 16), Image.BICUBIC), transforms.ToTensor()])

# transform3 = transforms.Compose(
#     [transforms.Resize(
#         (16, 16), Image.BICUBIC), transforms.ToTensor()])

transform2 = transforms.Compose(
    [transforms.ToTensor()]
)


j = 0
k = 0
l = 0


p=10
q=10
r=10
folder_dir2 = "Demo_images/Gender"
for images in os.listdir(folder_dir2):
    if(j == p):
        break
    if (images.endswith(".jpg")):
        print(images)
    path = folder_dir2+"/"+str(images)
    img = Image.open(path)
    image_tensor = Variable(transform(img)).to(device).unsqueeze(0)
    original_img_tensor = Variable(transform2(img)).to(device).unsqueeze(0)

    with torch.no_grad():
        sr_image1 = generator1(image_tensor)
        sr_image2 = generator2(image_tensor)
        save_image(torch.cat((sr_image1, sr_image2, original_img_tensor)),
                   "./images/output_gender/"+str(images))
        j += 1


folder_dir3 = "Demo_images/Eye"
for images in os.listdir(folder_dir3):
    if(k == q):
        break
    if (images.endswith(".jpg")):
        print(images)
    path = folder_dir3+"/"+str(images)
    img = Image.open(path)
    image_tensor = Variable(transform(img)).to(device).unsqueeze(0)
    original_img_tensor = Variable(transform2(img)).to(device).unsqueeze(0)
    with torch.no_grad():
        sr_image1 = generator1(image_tensor)
        sr_image3 = generator3(image_tensor)
        cc = torch.cat((sr_image1, sr_image3, original_img_tensor))
        save_image(cc,
                   "./images/output_eye/"+str(images))
        k += 1

folder_dir4 = "Demo_images/Eye_Hat/Eye"
for images in os.listdir(folder_dir4):
    if(l == r):
        break
    if (images.endswith(".jpg")):
        print(images)
    path = folder_dir4+"/"+str(images)
    img = Image.open(path)
    image_tensor = Variable(transform(img)).to(device).unsqueeze(0)
    original_img_tensor = Variable(transform2(img)).to(device).unsqueeze(0)
    with torch.no_grad():
        sr_image1 = generator1(image_tensor)
        sr_image4 = generator4(image_tensor)
        save_image(torch.cat((sr_image1, sr_image4, original_img_tensor)),
                   "./images/output_eye_hat/"+str(images))
        l += 1
