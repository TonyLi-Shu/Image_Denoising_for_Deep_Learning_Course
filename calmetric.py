import numpy as np
import os
from tqdm import tqdm
import sklearn
from PIL import Image
import json

DATE = "20230331"
read_path = "C:/Users/sli248/ZReconstruction/DeepLearning/Dataset/"+DATE+"Dataset/test/input"
label_path = "C:/Users/sli248/ZReconstruction/DeepLearning/Dataset/"+DATE+"Dataset/test/label"
pred_path = "C:/Users/sli248/ZReconstruction/DeepLearning/Unet-Segmentation-Pytorch-Nest-of-Unets/model/gen_images"

def cal_PSNR(image, image_hat):
    log_ = np.log10(np.max(image)**2/(np.mean(image-image_hat)**2))
    return 10*log_

def cal_RMSE(image, image_hat):

    return np.sqrt(np.sum((image - image_hat)**2)/(image.shape[0]*image.shape[1]))

def dict2arr(input_dict):
    All_list = []
    for key, value in input_dict.items():
        All_list.append(value)
    return np.array(All_list)


def cal_all(read_path, label_path):
    Metric_dict = dict()
    hist_metric = []
    #i = 0
    for file in tqdm(os.listdir(label_path),position=0,leave=True):
        name,suffix = os.path.splitext(file)
        if suffix == ".png":
            y_hat = np.asarray(Image.open(os.path.join(read_path,file)).convert('L').resize((256, 256)))
            y = np.asarray(Image.open(os.path.join(label_path, file)).convert('L'))
            y_hat = y_hat/y_hat.max()
            y  = y / y.max()
            snr_ = cal_PSNR(y,y_hat)
            rmse_ = cal_RMSE(y,y_hat)
            Metric_dict[name] = [snr_,rmse_]
            hist_metric.append([snr_,rmse_])
            # if i == 2:
            #     break
            # i += 1
    return Metric_dict,hist_metric

# Test it on the gt outpu
M_dict,hist_metric = cal_all(read_path, label_path)
M_arr = dict2arr(M_dict)

with open(os.path.join(os.path.dirname(pred_path),"test_input_output.txt"), 'w') as convert_file:
    convert_file.write(json.dumps(M_dict))

print("\ntest on the input")
print("the mean of snr and rmse:")
print(M_arr.mean(axis=0))


# Test it on the pred
print("\ntest on the pred")
pre_dict,hist_metric = cal_all(pred_path, label_path)
pre_arr = dict2arr(pre_dict)

with open(os.path.join(os.path.dirname(pred_path),"test_pred_output.txt"), 'w') as convert_file:
    convert_file.write(json.dumps(pre_dict))

print("the mean of snr and rmse:")
print(pre_arr.mean(axis=0))
# M_dict = cal_all(read_path, label_path)
# M_arr = dict2arr(M_dict)
#
# print("the mean of snr and rmse:")
# print(M_arr.mean(axis=0))


