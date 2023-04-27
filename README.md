# SUNet + UNet + SWINIR

For the Deep Learning course

1) **UNet** - U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

2) **SUNet** - Swin-Transformer UNet 
https://ieeexplore.ieee.org/document/9937486

3) **SwinIR** - Swin-Transformer Image Restoration
https://arxiv.org/abs/2108.10257

With Layer Visualization

## 1. Data set process

Find data set from the DIV2K and Flickr2K for training set 
BSD100 BSD200 and General100 for test set

## 2. Requirements

```
python>=3.6
torch==1.7.1
torchvision
torchsummary
tensorboardx
natsort
numpy
pillow
scipy
scikit-image
sklearn
```


## 3. Run the file

Add all your folders to this line 91-96 in the pytorch_run.py or pytorch_run_SUNet.py
```
t_data = '' # Input data
l_data = '' #Input Label
test_image = '' #Image to be predicted while training
test_label = '' #Label of the prediction Image
test_folderP = '' #Test folder Image
test_folderL = '' #Test folder Label for calculating the Dice score
 ```
models are writed in the SUNet.py SWINIR.py and Models.py

The model weights will be saved to ./model/UNet_XX_XX/best.pth
 
Run test set using eval.py (PATCH_IT.py is used for divide the image input patch for test set)

The output test will be saved to ./model/gen_images/

Run the difference map using AddingColorbarToTest.py

Run the Histogram_of_performance.ipynb to get the PSNR and SSIM

## 5. Visualization

To plot the loss , Visdom would be required. The code is already written, just uncomment the required part.
Gradient flow can be used too. Taken from (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)

A model folder is created and all the data is stored inside that.
Last layer will be saved in the model folder. If any particular layer is required , mention it in the line 361.


## 6. Results

Please refer to the report

## 7. Acknowledgment

Special acknowledgement to Fan and Liang who is the official designer for the SUNet and SwinIR. We made all the changes based on the official SUNet code. Official github link of SUNet and SwinIR is listed below:

https://github.com/FanChiMao/SUNet

https://github.com/JingyunLiang/SwinIR



