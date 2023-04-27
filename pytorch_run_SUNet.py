from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from PATCH_IT import patch_back,patch_img
import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net,U_Net_Perceptual,SCUNet
from SUNet import SUNet
from SWINIR import SwinIR
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
import pdb
#from ploting import VisdomLinePlotter
#from visdom import Visdom


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")


batch_size = 32
print('batch_size = ' + str(batch_size))

valid_size = 0.15

epoch = 600
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 0
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

"""
You also need to change the width and height in the Data_loader.py!!

"""


pin_memory = False
if train_on_gpu:
    pin_memory = True


InputW = 128
InputH = 128
window_size = 8

# model_test = SUNet(img_size=128, patch_size=4, in_chans=3, out_chans=3,
#                   embed_dim=96, depths=[8, 8, 8, 8],
#                   num_heads=[8, 8, 8, 8],
#                   window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
#                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                   norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                   use_checkpoint=False, final_upsample="Dual up-sample")

model_test = SwinIR(upscale=1, img_size=(InputH, InputW),
                   window_size=window_size, img_range=1., depths=[2, 2, 2, 2],
                   embed_dim=30, num_heads=[2, 2, 2, 2], mlp_ratio=1, upsampler='pixelshuffledirect')

DATE = "20230331"

t_data = "C:/Users/sli248/ZReconstruction/DL/dataset/train/input/"
l_data = "C:/Users/sli248/ZReconstruction/DL/dataset/train/label/"
test_image = "C:/Users/sli248/ZReconstruction/DL/dataset/test/input/BSDS100_38092.png"
test_label = "C:/Users/sli248/ZReconstruction/DL/dataset/test/label/BSDS100_38092.png"
test_folderP = "C:/Users/sli248/ZReconstruction/DL/dataset/test/input/*"
test_folderL = "C:/Users/sli248/ZReconstruction/DL/dataset/test/label/*"

InputW = 128
InputH = 128

model_test.to(device)
torchsummary.summary(model_test, input_size=(3, InputW, InputH))

def self_transforms(inputW, inputH):
    if inputW == 0:
        return torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
    else:
        return torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(inputW),
                #torchvision.transforms.Resize((inputW, inputH)),
                torchvision.transforms.ToTensor()
            ])

train_transform = self_transforms(InputW, InputH)
test_transform = self_transforms(0, 0)


Training_Data = Images_Dataset_folder(t_data,
                                      l_data,
                                      train_transform,
                                      train_transform)


num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
print(len(train_idx), len(valid_idx))


train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)



# for x, y in train_loader:
#     x, y = x.to(device), y.to(device)
#     print("X = ",x.min(),x.max())
#     print("Y = ",y.min(),y.max())



initial_lr = 1e-4 #1e-3
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD weight_decay=0.1
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)


New_folder = './model'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
#Setting the folder of saving the predictions
#######################################################

read_pred = os.path.join(New_folder,'pred')

#######################################################
#Checking if prediction folder exixts
#######################################################

if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

#######################################################
#checking if the model exists and if true then delete
#######################################################

model_state_folder = 'UNet_D_' + str(epoch) + '_' + str(batch_size)
read_model_path = os.path.join(New_folder,model_state_folder)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

#######################################################
#Training loop
#######################################################

def L1_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss
def L2_loss(output, target):
    loss = torch.mean(torch.square(output - target))
    return loss

#def perceptual_loss(output, target):


calc_loss = torch.nn.MSELoss() #L2_loss
# calc_loss = torch.nn.L1Loss()
for i in range(epoch):
    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()

    lr = scheduler.get_last_lr() #get_lr()
    #######################################################
    #Training Data
    #######################################################
    model_test.train()
    k = 1
    for x, y in tqdm(train_loader,position=0,leave=True):
        # print(x.shape,y.shape)
        # print(x.size(), y.size())
        x, y = x.to(device), y.to(device)


        # ====== Perceptual Loss ======
        y_pred = model_test(x)
        opt.zero_grad()

        loss =  calc_loss(y_pred, y)
        train_loss += loss.item() * x.size(0)


        loss.backward()
        opt.step()


    scheduler.step()

    #######################################################
    #Validation Step
    #######################################################

    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory

    for x1, y1 in tqdm(valid_loader,position=0,leave=True):
        x1, y1 = x1.to(device), y1.to(device) #[batch_size, channel, w, h]

        y_pred1 = model_test(x1)
        loss = calc_loss(y_pred1, y1)     # Dice_loss Used

        valid_loss += loss.item() * x1.size(0)



    #######################################################
    # Saving the predictions
    #######################################################

    im_tb = Image.open(test_image)
    im_label = Image.open(test_label)
    s_tb = test_transform(im_tb)
    ch_n, h, w = s_tb.size()
    stack_s_tb = patch_img(s_tb.unsqueeze(0), resizeL=InputW).to(device)
    #print("stack_s_tb")
    # print(stack_s_tb.size())
    # pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    # print( stack_s_tb.size(0) , batch_size)
    pred_tb_all = None
    step_size = 12
    if stack_s_tb.size(0) > step_size:
        for slice_ch_n in range(0,stack_s_tb.size(0)+step_size,step_size):

            if slice_ch_n < stack_s_tb.size(0):
                if slice_ch_n +step_size>= stack_s_tb.size(0):
                    stack_s_tb_ = stack_s_tb[slice_ch_n:, :, :, :]
                else:
                    stack_s_tb_ = stack_s_tb[slice_ch_n:slice_ch_n+step_size, :, :, :]
                #print(slice_ch_n)
                #print(stack_s_tb_.shape)
                pred_tb_ = model_test(stack_s_tb_).cpu()
                if pred_tb_all is None:
                    pred_tb_all = pred_tb_
                else:
                    pred_tb_all = torch.cat([pred_tb_all, pred_tb_], dim=0)
    else:
        pred_tb_all = model_test(stack_s_tb.to(device))

    #pred_tb = model_test().cpu()  # model_test().cpu()

    pred_tb = pred_tb_all.cpu().detach().numpy()  # [0]#.transpose(2, 0, 1)

    back_img = patch_back(pred_tb, h, w)
    #back_img = (back_img - back_img.min())/ (back_img.max()-back_img.min())
    # pdb.set_trace()
    # pred_tb = threshold_predictions_v(pred_tb)
    fig = plt.figure()
    plt.imsave(
        './model/pred/img_iteration_' + str(n_iter) + '_epoch_'
        + str(i) + '.png', back_img.transpose(1, 2, 0))
    plt.close(fig)

    train_loss = train_loss  # / len(train_idx)
    valid_loss = valid_loss  # / len(valid_idx)
    lossT.append(train_loss)
    lossL.append(valid_loss)
    if (i + 1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
    if (i + 1) % 10 == 0:
        savelossT = np.array(lossT)
        savelossL = np.array(lossL)
        np.save("./model/train_loss.npy", savelossT)
        np.save("./model/valid_loss.npy", savelossL)
        torch.save(model_test.state_dict(),
                   os.path.join(read_model_path, 'Unet_epoch_' + str(i) + '_batchsize_' + str(batch_size) + '.pth'))


    # #####################################
    # # for kernals
    # #####################################
    # x1 = torch.nn.ModuleList(model_test.children())
    # # x2 = torch.nn.ModuleList(x1[16].children())
    # # x3 = torch.nn.ModuleList(x2[0].children())
    #
    # # To get filters in the layers
    # # plot_kernels(x1.weight.detach().cpu(), 7)
    #
    # #####################################
    # # for images
    # #####################################
    # x2 = len(x1)
    # dr = LayerActivations(x1[x2 - 1])  # Getting the last Conv Layer
    #
    # img = Image.open(test_image)
    # s_tb = test_transform(img)
    #
    # pred_tb = model_test(s_tb.unsqueeze(0).to(device))[0].cpu()
    # # pred_tb = torch.sigmoid(pred_tb)
    # pred_tb = pred_tb.detach().numpy()
    #
    # # plot_kernels(dr.features, n_iter, 7, cmap="rainbow")

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

    if valid_loss <= valid_loss_min:  # and epoch_valid >= i: # and i_valid <= 2:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(), os.path.join(read_model_path, 'best.pth'))
        # print(accuracy)
        #  if round(valid_loss, 4) == round(valid_loss_min, 4):
        #      print(i_valid)
        #      i_valid = i_valid+1
        i_valid = 0
        valid_loss_min = valid_loss
    else:
        i_valid += 1
        if i_valid > 100:
            print("due to the loss is not decreasing from last 50 epochs, the model stop training")
            break

#######################################################
# generate the loss plot
#######################################################
fontsize = 20
fig = plt.figure(figsize=(12,8))
plt.plot(lossT,label = "train loss")
plt.plot(lossL,label = "valid loss")
plt.title("Loss of training and validation", fontsize = fontsize)
plt.xlabel("Epoch",fontsize = fontsize)
plt.ylabel("Loss", fontsize = fontsize)
plt.legend(fontsize = fontsize)
plt.grid()
plt.savefig("./model/loss_plot.png",bbox_inches='tight')
plt.show()
plt.close(fig)
#writer1.close()

#######################################################
#if using dict
#######################################################

#model_test.filter_dict
