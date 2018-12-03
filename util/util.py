from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import h5py

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor2im2(inp, imtype=np.uint8):
    if isinstance(inp, torch.Tensor):
        inp = inp.data
    else:
        return inp

    inp = inp[0].cpu().float().numpy()
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)*255
    return inp.astype(imtype)

def tensor2im3(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    image_numpy =  np.clip(image_numpy, 0, 1)*255

    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dnd_srgb(data_folder):
        data = []
        '''
        Utility function for denoising all bounding boxes in all sRGB images of
        the DND dataset.

        denoiser      Function handle
                      It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                      and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
        data_folder   Folder where the DND dataset resides
        out_folder    Folder where denoised output should be written to
        '''

        # load info
        infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
        info = infos['info']
        bb = info['boundingboxes']
        #print('info loaded\n')
        # process data
        for i in range(50):
            filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
            img = h5py.File(filename, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)
            # bounding box
            ref = bb[0][i]
            boxes = np.array(info[ref]).T
            for k in range(20):
                idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
                Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
                data.append(Inoisy_crop)
        return data


class Noise():
    def __init__(self,img,noise_model=0):
        self.noise_model = noise_model
        self.img = np.array(img)

    def gaussian_noise(self,min_stddev=10,max_stddev=50):
        noise_img = self.img.astype(np.float)
        stddev = np.random.uniform(min_stddev, max_stddev)
        noise = np.random.randn(*self.img.shape) * stddev
        noise_img += noise
        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np.uint8(noise_img))

    def add_impulse_noise(self,min_occupancy=5, max_occupancy=45):
        occupancy = np.random.uniform(min_occupancy, max_occupancy)
        mask = np.random.binomial(size=self.img.shape, n=1, p=occupancy / 100)
        noise = np.random.randint(256, size=self.img.shape)
        img = self.img * (1 - mask) + noise * mask
        return Image.fromarray(np.uint8(img))

    def add_poisson_noise(self):
        vals = len(np.unique(self.img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(self.img * vals) / float(vals)
        return Image.fromarray(np.uint8(noisy))

    def add_speckle_noise(self):
        row, col, ch = self.img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = self.img + self.img * gauss
        return Image.fromarray(np.uint8(noisy))

    def add_signal_dependant_noise(self,min_stddev=0.01,max_stddev=0.23):
        noise_img = self.img.astype(np.float)/255

        stddev = np.random.uniform(min_stddev, max_stddev)
        stddev_ = np.random.uniform(min_stddev, max_stddev)

        noise_map_s = noise_img*stddev_
        noise_s = np.random.randn(*self.img.shape) *noise_map_s
        noise_map_c = stddev
        noise_c = np.random.randn(*self.img.shape) *noise_map_c

        noise_img += noise_c
        noise_img += noise_s

        noise_img = np.clip(noise_img*255, 0, 255).astype(np.uint8)
        map = np.clip((noise_map_s + noise_map_c)*255, 0, 255).astype(np.uint8)
        #plt.imshow(map)
        #plt.show()
        return Image.fromarray(noise_img),Image.fromarray(map)

    def run(self):
        if(self.noise_model==0):
            return self.gaussian_noise()
        if(self.noise_model==1):
            return self.add_impulse_noise()
        if(self.noise_model==2):
            return self.add_poisson_noise()
        if(self.noise_model==3):
            return self.add_speckle_noise()
        if(self.noise_model==4):
            return self.add_signal_dependant_noise()
