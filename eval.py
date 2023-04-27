from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F
import pdb
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from TUNet import U_Transformer
import shutil
import random
from SWINIR import SwinIR
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, U_Net_Perceptual
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
from tqdm import tqdm
from Transformer_UNet import Transformer_U_Net
#from ploting import VisdomLinePlotter
#from visdom import Visdom
#from pytorch_run import self_transforms
from SUNet import SUNet
from PATCH_IT import patch_back,patch_img

def self_transforms():
   return torchvision.transforms.Compose([
                #torchvision.transforms.Resize((224,224)),
                #torchvision.transforms.CenterCrop(96),
                torchvision.transforms.ToTensor()#,
                #torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
test_folderP = "C:/Users/sli248/ZReconstruction/DL/dataset/test/input/*"
test_folderL = "C:/Users/sli248/ZReconstruction/DL/dataset/test/label/*"
pth_model_path = "C:/Users/sli248/ZReconstruction/"+ \
                "DL/Unet-Segmentation-Pytorch-Nest-of-Unets/model/"+ \
                "Unet_D_600_32/best.pth"
epoch = 10
upper_ = 0.015
lower_ = -0.015

InputW = 128
InputH = 128

data_transform = self_transforms()
# data_transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((512,512)),
#             #torchvision.transforms.CenterCrop(96),
#             torchvision.transforms.RandomRotation((-10,10)),
#             #torchvision.transforms.Grayscale(),
#             torchvision.transforms.ToTensor()#,
#             #torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])


model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

# For UNet
# model_test = model_unet(model_Inputs[0], 3, 3)
# For SUNet
# model_test = SUNet(img_size=128, patch_size=4, in_chans=3, out_chans=3,
#                   embed_dim=96, depths=[8, 8, 8, 8],
#                   num_heads=[8, 8, 8, 8],
#                   window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
#                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                   norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
#                   use_checkpoint=False, final_upsample="Dual up-sample")
# For SWINIR
window_size = 8
model_test = SwinIR(upscale=1, img_size=(InputH, InputW),
                   window_size=window_size, img_range=1., depths=[2, 2, 2, 2],
                   embed_dim=30, num_heads=[2, 2, 2, 2], mlp_ratio=1, upsampler='pixelshuffledirect')
#######################################################
#checking if cuda is available
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Loading the model
#######################################################

model_test.load_state_dict(torch.load(pth_model_path))
model_test.to(device)
model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
#######################################################

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort


#read_test_folder112 = './model/gen_images'
read_test_folder112 = os.path.join(os.path.dirname(os.path.dirname(pth_model_path)),'gen_images')


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

#read_test_folder_P_Thres = './model/pred_threshold'
read_test_folder_P_Thres = os.path.join(os.path.dirname(os.path.dirname(pth_model_path)),'pred_threshold')


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

#read_test_folder_L_Thres = './model/label_threshold'
read_test_folder_L_Thres = os.path.join(os.path.dirname(os.path.dirname(pth_model_path)),'label_threshold')

if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)




#######################################################
#saving the images in the files
#######################################################

img_test_no = 0
model_test.eval()
for i in tqdm(range(len(read_test_folder)),position=0, leave=True):
    im = Image.open(x_sort_test[i])
    # x_sort_test[i] = C:/Users/sli248/ZReconstruction/DeepLearning/Dataset/20230302Dataset/test/input\female_1_60.png
    base_file_name_ = os.path.basename(x_sort_test[i])
    s = data_transform(im)#.to(device) #shape = [1, 1, 512, 512]

    ch_n, h, w = s.size()
    stack_s_tb = patch_img(s.unsqueeze(0), resizeL=InputW)
    pred_tb_all = None
    step_size = 12
    if stack_s_tb.size(0) > step_size:
        for slice_ch_n in range(0, stack_s_tb.size(0) + step_size, step_size):

            if slice_ch_n < stack_s_tb.size(0):
                if slice_ch_n + step_size >= stack_s_tb.size(0):
                    stack_s_tb_ = stack_s_tb[slice_ch_n:, :, :, :]
                else:
                    stack_s_tb_ = stack_s_tb[slice_ch_n:slice_ch_n + step_size, :, :, :]
                # print(slice_ch_n)
                # print(stack_s_tb_.shape)
                stack_s_tb_ = stack_s_tb_.to(device)
                pred_tb_ = model_test(stack_s_tb_)#.cpu()
                if pred_tb_all is None:
                    pred_tb_all = pred_tb_
                else:
                    pred_tb_all = torch.cat([pred_tb_all, pred_tb_], dim=0)
    else:
        pred_tb_all = model_test(stack_s_tb.to(device))


    pred_tb = pred_tb_all.detach().cpu().numpy()
    pred = patch_back(pred_tb, h, w)

    # #pred = model_test(s)#.cpu()
    # pred = model_test(s)[0].cpu()
    # #pred = torch.sigmoid(pred)
    # pred = pred.detach().numpy()

    if i % 24 == 0:
        img_test_no = img_test_no + 1

    #fig = plt.figure(figsize=(8, 8))
    #plt.imshow(pred.transpose(1, 2, 0))
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.savefig(os.path.join(read_test_folder112,base_file_name_),bbox_inches='tight')
    # plt.close(fig)
    fig = plt.figure()
    plt.imsave(os.path.join(read_test_folder112,base_file_name_), pred.transpose(1, 2, 0))
    plt.close(fig)
    #break
    # print(pred[0][0].max(),pred[0][0].min()) max=0.9983 min=0.522
    # x1 = plt.imsave('./model/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
    #                 + '_img_no_' + str(img_test_no) + '.png', pred[0][0],cmap='gray')

####################################################
#Calculating the Dice Score
####################################################

#
# read_test_folderP = glob.glob('./model/gen_images/*')
# x_sort_testP = natsort.natsorted(read_test_folderP)
#
#
# read_test_folderL = glob.glob(test_folderL)
# x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort
#
#
# dice_score123 = 0.0
# x_count = 0
# x_dice = 0
#
# for i in range(len(read_test_folderP)):
#
#     x = Image.open(x_sort_testP[i])
#     s = data_transform(x)
#     s = np.array(s)
#     s = threshold_predictions_v(s)
#
#     #save the images
#     # plt.figure(figsize=(8, 8))
#     # plt.imshow(s, vmin=0,vmax=255,camp='gray')
#     # plt.colorbar()
#     # plt.savefig('./model/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
#     #                 + '_img_no_' + str(img_test_no) + '.png')
#     x1 = plt.imsave('./model/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
#                     + '_img_no_' + str(img_test_no) + '.png', s)
#
#     y = Image.open(x_sort_testL[i])
#     s2 = data_transform(y)
#     s3 = np.array(s2)
#    # s2 =threshold_predictions_v(s2)
#
#     #save the Images
#     # plt.figure(figsize=(8, 8))
#     # plt.imshow(s3, vmin=0,vmax=255,camp='gray')
#     # plt.colorbar()
#     # plt.savefig('./model/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
#     #                 + '_img_no_' + str(img_test_no) + '.png')
#     y1 = plt.imsave('./model/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
#                     + '_img_no_' + str(img_test_no) + '.png', s3)
#     total = dice_coeff(s, s3)
#     print(total)
#
#     if total <= 0.3:
#         x_count += 1
#     if total > 0.3:
#         x_dice = x_dice + total
#     dice_score123 = dice_score123 + total
#
#
# print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))


#print(x_count)
#print(x_dice)
#print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))