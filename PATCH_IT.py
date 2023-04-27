import os
import numpy as np
from PIL import Image
from torch import optim
import torch.utils.data
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort


def patch_img(img, resizeL, step_ratio = 1):
    """
    :param img: batch=1,channel=3,w,h
    :return: img: batchs=N, channel=3, cropw, cropg
    """
    bs, n_ch, h, w = img.size()
    # print(img.max(),img.min())
    step = int(resizeL * step_ratio)
    stack = None
    n_w = w // step + 1
    n_h = h // step + 1
    # print(w,resizeL, n_w, h,n_h)

    # plt.figure()
    # f, axarr = plt.subplots(n_h,n_w)
    for w_i in range(0, n_w*step+1, step):
        for h_i in range(0, n_h*step+1, step):
            patch = torch.zeros(1,n_ch,resizeL,resizeL)
            if w_i +resizeL >= w:
                w_crop = w
                try:
                    patch[:, :, -resizeL:, -resizeL:] = img[:, :n_ch, -resizeL:, -resizeL:]
                except:
                    pass
            else:
                w_crop = w_i+resizeL
            if h_i + resizeL >= h:
                h_crop = h
                try:
                    patch[:, :, -resizeL:, -resizeL:] = img[:, :n_ch, -resizeL:, -resizeL:]
                except:
                    pass
            else:
                h_crop = h_i + resizeL
            #print(h_i,h_crop, w_i, w_crop)
            if h_crop-h_i > 0 and w_crop-w_i > 0:
                patch[:,:,:h_crop-h_i,:w_crop-w_i] = img[:,:,h_i:h_crop,w_i:w_crop]
                if stack is None:
                    stack = patch
                else:
                    stack = torch.concat([stack,patch],dim=0)
                # print(stack.size())
                #plot_w = w_i//resizeL
                #plot_h = h_i//resizeL
                #npimg = patch.detach().numpy().transpose(1,2,0)*255
                #IMG = Image.fromarray(npimg.astype('uint8'))
                #axarr[plot_h,plot_w].imshow(IMG)
    return stack
    # plt.show()

def patch_back(stack, h, w, step_ratio = 1):
    """
    :param stack: numpy shape
    :param h: original height the second last one
    :param w: original width the last one
    :return: the fix image
    """
    N, ch_n, resizeL,resizeL = stack.shape
    #print(stack.shape)
    step = int(resizeL * step_ratio)
    n_w = w // step + 1
    n_h = h // step + 1
    back_img = np.zeros((3,(n_h+1)*resizeL,(n_w+1)*resizeL+step))
    #print(back_img.shape)
    stack_i = 0

    for w_i in range(n_w):
        for h_i in range(n_h):
            if stack_i < N:
                w_s = w_i * step
                h_s = h_i * step
                if w_i == n_w -1  and h_i != 0:
                    back_img[:,h_s:h_s+step,w_s:w_s+step] = stack[stack_i, :, -step:,:step]
                elif h_i == 0 and w_i != 0:
                    back_img[:,h_s:h_s+step,w_s:w_s+step] = stack[stack_i,:,:step,-step:]
                elif h_i == 0 and w_i == 0:
                    back_img[:,h_s:h_s+step,w_s:w_s+step] = stack[stack_i,:,:step,:step]
                else:
                    back_img[:,h_s:h_s+step,w_s:w_s+step] = stack[stack_i, :, -step:, -step:]
            stack_i += 1
            #print(f"back_img[:,{h_s}:{h_s+step},{w_s}:{w_s+step}] = stack[{stack_i},:,{-step}:,{-step}:]")
    # for w_i in range(0, n_w*resizeL+1, step):
    #     for h_i in range(0, n_h*resizeL+1, step):
    #         if (h_i + step) <= n_h*resizeL and \
    #                 (w_i + step) <= n_w*resizeL and \
    #                 stack_i< stack.shape[0]:
    #             print(f"back_img[:,{w_i}:{w_i+step},{h_i}:{h_i+step}] = stack[{stack_i},:,{-step}:,{-step}:]")
    #             #print(h_i, h_i+resizeL - step, w_i, (w_i + resizeL - step))
    #             back_img[:,w_i:w_i+step,h_i:h_i+step] = stack[stack_i,:,-step:,-step:]#.unsqueeze(0)
    #             stack_i += 1
    #print("final image = ",h,w)
    return back_img[:,:h,:w]