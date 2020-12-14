# Implementation of Super-Resolution Generative Adversarial Network(SRGAN) on CelebA 

The dataset will be CelebA, which can be download at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

To run the program on colab, please download the dataset and upload locally.

The model we use is based on the 
*"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"*
https://arxiv.org/abs/1609.04802


## Summary
In this project, We will look at how to generate high-resolution images from low-resolution ones by using SRGAN. SRGAN is the first framework capable of inferring photo-realistic natural images for 4× upscaling factors. In this project, we will show how well it can be applied in the CelebA dataset. Also, we will take more challenging task with  8x upscalling task. 

## Method
  1. **Data preprocessing**

Our goal is to train a model which can infers the high-resolution(HR) images given the low-resolution(LR) images. 

Before the traning, we divide our input images with 176x176 into two parts: HR images and LR images and then use center crop to 144x144 to focus on the face.

For HR images, we resize the 176x176 input images to 256x256 images which are set as original HR images. For LR images, we resize the 256x256 input images to 64x64 ones with scalar 4. 

For both LR and HR images, we normalize them with `(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)` because later the dataset will be fed into VGG19 pretraning model. 

2. **Adversarial network architecture**

Following the "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network". SRGAN just like a ordinary GAN, has one Generator and one Discriminator, what they do is to find their own maximum profit (lowest loss) to achieve a state (like Nash Equilibria) that the generator can generate the HR images to fool the discriminator. The architecture can be seen at figure 1.

![pic](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/0D1EB002-112A-4400-A63B-511F88F4152C.png)

*Figure 1: Architecture of Generator and Discriminator Network with corresponding kernel size (k), number of feature maps
(n) and stride (s) indicated for each convolutional layer. [1]*

**3. Loss Function** 

 

 - Generator loss

Generator loss in SRGAN is also known as perceptual loss. It can be defined as 
 ![Perceptual loss](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/image.png)







