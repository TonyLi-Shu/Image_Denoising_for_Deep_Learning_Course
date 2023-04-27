from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import natsort
import shutil
import cv2


test_folderP = "C:/Users/sli248/ZReconstruction/DL/dataset/test/input/*"
test_folderL = "C:/Users/sli248/ZReconstruction/DL/dataset/test/label/"

test_folderO = "C:/Users/sli248/ZReconstruction/"+ \
                "DL/Unet-Segmentation-Pytorch-Nest-of-Unets/model/"+ \
                "gen_images/"

save_folder = os.path.join(os.path.dirname(test_folderO),"../colorbar")
save_difffolder = os.path.join(os.path.dirname(test_folderO),"../colorbar_diff")

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if not os.path.exists(save_difffolder):
    os.mkdir(save_difffolder)
def load_image_01(img_path, low_=0, high_=255):
    image = Image.open(img_path).convert('L')
    #image = cv2.imread(img_path,-1)
    #print(image)
    # im = np.load(x_sort_test[i])
    image = np.asarray(image).copy()
    image[image <= low_] = low_
    image[image >= high_] = high_
    image = (image - image.min()) / (image.max() - image.min())
    return image

def plot_image(image, foldername, save_folder):
    fig = plt.figure()
    plt.imshow(image, vmin=0, vmax=1, cmap="gray")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(save_folder, file_name_ +"_"+ foldername+".png"), bbox_inches='tight')
    plt.close(fig)

def plot_different(input, output, label, save_folder, vmin = 0, vmax= 0.2):
    diff_label_input  = np.abs(label - input)
    diff_label_output = np.abs(label - output)

    fig = plt.figure()
    plt.imshow(diff_label_input, vmin=vmin, vmax=vmax, cmap="plasma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(save_folder, file_name_ + "_label_input.png"), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(diff_label_output, vmin=vmin, vmax=vmax, cmap="plasma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(save_folder, file_name_ + "_label_output.png"), bbox_inches='tight')
    plt.close(fig)


read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)

Low_ = 0#0.05
High_ = 255#0.95
for i in tqdm(range(len(read_test_folder)),position=0, leave=True):
    base_file_name_ = os.path.basename(x_sort_test[i])
    file_name_,_ = os.path.splitext(base_file_name_)
    label_path = os.path.join(test_folderL,base_file_name_)
    input_path = x_sort_test[i]
    output_path = os.path.join(test_folderO,base_file_name_)
    input = load_image_01(input_path, Low_, High_)
    label = load_image_01(label_path, Low_, High_)
    output = load_image_01(output_path, Low_, High_)

    #plot_image(input, "input", save_folder)
    #plot_image(label, "label", save_folder)
    #plot_image(output, "output", save_folder)

    plot_different(input, output, label, save_difffolder)
    #break
    #print(base_file_name_)
    #print(input.max(), input.min())
    #break


#plt.close(fig)
#fig = plt.figure()
#plt.imsave(os.path.join(read_test_folder112,base_file_name_), show_pred_,vmin=0,vmax=1,cmap='gray')
#

