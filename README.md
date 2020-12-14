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

![pic](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/epoch_3.jpeg)

*Figure 1: Architecture of Generator and Discriminator Network with corresponding kernel size (k), number of feature maps
(n) and stride (s) indicated for each convolutional layer. [1]*

**3. Loss Function** 

 - Generator loss

Generator in SRGAN use perceptual loss. The perceptual loss can be defined as:
 ![Perceptual loss](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/image.png)


The perceptual loss is the weighted sum of a content loss and an adversarial loss. 

 - Content Loss

In the SRGAN,  the content loss is basically the pixel-wise MSE loss between the last feature maps of a pre-trained VGG network from the LR and HR images. It can be calculated as:

![vgg content loss](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/3DE85850-7AE8-4AD2-9686-C41CCE054DD5.png)

Here W and H just the dimensions of the respective feature maps.

In our work, however, we use L1 loss function instead of MSE loss funtion.

 - Adversarial Loss

The adversarial loss in SRGAN is generative loss in the GAN,the loss function is:

![adversarial loss](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/BAE81DFD-A84D-45E4-B55C-C75B23DA0F35.png)

In our project, we use  least square loss function to calculte the generator and discriminator loss.
The generator loss is:

![g loss](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/F839CAF9C9B171FE2CFFDCEACE761BAC.jpg)

and the discriminator loss is:

![d loss](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/BBFCB69961466E7AC17FDB9B3D2BCB7F.jpg)

For optimization, we use adam optimizer with learning rate 1e-5.

## Experiment

For the purpose of having the same color as original preprocessed HR images, we also use Tanh activation function after last convolution and scale it with 2.38 and add bias 0.26 in generator network to match the original HR images range.

To better evaluate the performance, we also use  bicubic interpolation (popular method for improving image resolution) for comparison. Therefore the output images are LR images, bicubic images, SRGAN generated images, original HR images.

We finally train 3 epochs with 4x and 8x upscalling factor on 20,000 images from CelebA. It costs about 24 hours by using  NVIDIA GPU on Colab. The generated HR images look pretty similar with the original HR images. We run 3 epoch on such large datasets because it can reduce the checkboard effect if we take more images into training.

## Result and Example

At first, we use 4x upscalling factor and just after few batches, it shows good results. 

![enter image description here](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/epoch_1_4scaler.jpeg)

Figure 2. *The images from left to right is **LR images,  bicubic images, SRGAN generated images,  original HR images** respectively*

However, when we want to accomplish 8x upscalling. The images begins to have checkerboard effect.

![8 sclar](https://github.com/tjjj686/dl_project_srgan/blob/main/pic/6BE699B2-8B54-4B03-9790-2C2B42DA1A20.png)

