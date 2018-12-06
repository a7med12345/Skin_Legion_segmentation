# Skin_Legion_segmentation

## Data Preparation

Download Lesion segmentation dataset from:
[ISIC 2018 website](https://challenge2018.isic-archive.com/).

Dataset contains 2594 images and their corresponding groundtruth of size around 3000x3000.

Example of data:

![alt-text-1](images/ISIC_0000055.jpg "Image example") ![alt-text-2](images/ISIC_0000055_segmentation.png "Corresponding Groundtruth")
![alt-text-3](images/ISIC_0000056.jpg "Image example") ![alt-text-4](images/ISIC_0000056_segmentation.png "Corresponding Groundtruth")


* We apply a sliding window of size 512x512 with step size =350; on each image in the data and its
corresponding ground truth. 
* We keep only the cropped windows that have at least 30% foreground and at maximum 85% background.
* We select randomly a maximum of 9 samples from each image.

* To apply the transformation run:

`python data_processing.py --path_images /path/to/images --path_gt /path/to/groundtruth`

* Choose a subset of data and move it to testA and testB folders inside `./lesion_dataset`.

* Example of training data inputs (size 512x512):

![alt-text-5](images/0.jpg "Image example") ![alt-text-6](images/0_target.jpg "Corresponding Groundtruth")


</br>
</br>
</br>

</br>




## Experiments:

### Image to Image mapping

To train:

L2 loss: `python train.py --name seg_l2 --model lesion --batch_size 4 --dataroot ./datasets/lesion_dataset --gpu_ids 0`

BCE loss: `python train.py --name seg_bce --model lesion --batch_size 4 --dataroot ./datasets/lesion_dataset --gpu_ids 0 --loss_type bce`

1.Network architecture: UNet


2.Experiment with L2 Loss

 
2.1.  Training results:

2.1.1. Training curve

![alt-text-7](checkpoints/seg_l2/36d7e2eb89331e.svg "training_l2")

 2.1.2. Network Output Respectively after 200 epochs: 
 
 `Input image; Output segmentation; Target`
![alt-text-7](checkpoints/seg_l2/seg_l2_training.png "training_l2")

    
2.2. Testing results:
    
2.2.1. Output Images: 

We use [Otsu thresholding](http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html)
to generate the binary image from the network output.

Respectively: `Input image; Target; Output segmentation; True Positive; Wrongly classified`

![alt-text-10](images/l2_loss_0_5.png "Output")
![alt-text-10](images/l2_loss_2_5.png "Output")


2.2.2. Testing Loss:
        
![alt-text-9](images/MSE_loss.png "L2 loss")

 2.2.3. Testing Accuracy:

![alt-text-10](images/L2_accuracy.png "L2 accuracy")

        


3.Experiment with binary cross entropy loss

3.1.  Training results:

3.1.1. Training curve
![alt-text-7](checkpoints/seg_bce/36d7e66164e458.svg "training_l2")

3.1.2. Network Output Respectively after 200 epochs: 

`Input image; Output segmentation; Target`
![alt-text-7](checkpoints/seg_bce/seg_bce_training.png "training_l2")

3.2. Testing results:
    
3.2.1. Output Images: 

Similar to l2 loss; We use [Otsu thresholding](http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html)
to generate the binary image from the network output.

Respectively: `Input image; Target; Output segmentation; True Positive; Wrongly classified`

![alt-text-10](images/BCE_example_0_5.png "Output")
![alt-text-10](images/BCE_Example_1_5.png "Output")


3.2.2. Testing Loss:
        
![alt-text-9](images/BCE_loss.png "L2 loss")

3.2.3. Testing Accuracy:

![alt-text-10](images/bc3_accuracy.png "L2 accuracy")

4.Comparison between the two losses:

| Loss     | Accuracy |
| ---      | ---       |
| Mean Squared Error | 0.832071031842913       |
| Binary Cross entropy     | 0.8290215083530971       |

### Generative adversial network experiment

To train: 

`python train.py --name gan_seg_l2 --model lesion_gan --batch_size 4 --dataroot ./datasets/lesion_dataset --gpu_ids 0 --netD test`

1. Generator and Discriminator architecture:

    1.1. Generator: Unet like architecture
    
    1.2. Discriminator:
    ![alt-text-7](images/discriminator.png "Discriminator architecture")
    



##### Testing Results:

Respectively: `Input image; Target; Output segmentation; True Positive; Wrongly classified`

![alt-text-10](images/GAN_1.png "Output")
![alt-text-10](images/GAN_3.png "Output")


Testing accuracy:

![alt-text-10](images/GAN_accuracy.png "Output")

After 200 epochs accuracy of:

Test Accuracy: 0.8516359056745256

