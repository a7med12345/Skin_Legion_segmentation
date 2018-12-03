from __future__ import division
import os.path
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import os
from data.image_folder import make_dataset



class LesionDataset():


    def __init__(self,opt):
        self.opt =opt

        self.dir_A = opt.path_images
        self.dir_B = opt.path_gt

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)

        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)


        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        transform_ = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_2 = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_)
        self.transform2 = transforms.Compose(transform_2)

        self.crop_size = (512,512)





    def sliding_window(self,im, stepSize=350):
        windowSize = self.crop_size
        # slide a window across the image
        w, h = im.size
        for y in range(0, h-windowSize[1], stepSize):
            for x in range(0, w-windowSize[0], stepSize):
                # yield the current window
                #yield im[y:y + windowSize[1], x:x + windowSize[0]]
                #yield im.crop((x, y,  x+windowSize[0],y + windowSize[1]))
                yield x,y


    def test_black_white_ratio(self,x,thres1=0.3, thres2=0.85):

        #array = np.asarray(input)

        a = np.count_nonzero(x)
        if(a/x.size >thres1 and a/x.size <thres2):
            return True
        else:
            return False

    def return_images(self,list_points,img,img_label,a,b):
        data = []
        label = []
        dddd = []
        for j in range(0,len(list_points)):
            i = random.randint(0, len(list_points) - 1)
            while(i in dddd):
                i = random.randint(0,len(list_points)-1)
            dddd.append(i)
            x,y = list_points[i]
            im_crop = img.crop((x, y, x + a, y + b))
            label_crop = img_label.crop((x, y, x + a, y + b))

            test =self.test_black_white_ratio(np.asarray(label_crop))
            if(test):

                data.append(im_crop)
                label.append(label_crop)
            if(j>8):
                break

        return data,label



    def getdata(self, index):


        a, b = self.crop_size


        A_path = self.A_paths[index]
        B_path = self.B_paths[index]



        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')


        l = list(self.sliding_window(A_img))
        #x,y= l[index%len(l)]


        #A_img = A_img.crop((x, y, x + a, y + b))
        #B_img = B_img.crop((x, y, x + a, y + b))



        A, B = self.return_images(l,A_img,B_img,a,b)





        return {'A': A, 'B': B}

    def __len__(self):
        return self.A_size



if __name__ == '__main__':
    import argparse

    opt = argparse.ArgumentParser()
    opt.add_argument("--path_images", required=True,help="folder containing images")
    opt.add_argument( "--path_gt", required=True,help="folder containing corresponding groundtruth")
    opt = opt.parse_args()

    dataset = LesionDataset(opt)

    k = 0
    for i in range(0,dataset.A_size):

            input = dataset.getdata(i)
            pathA_train = "datasets/lesion_dataset/trainA/"
            pathB_train = "datasets/lesion_dataset/trainB/"
            if(len(input["A"])>0):
                for j in range(0,len(input["A"])):
                    name = pathA_train + str(k)+".jpg"
                    name2 = pathB_train + str(k) + ".png"
                    input["A"][j].save(name)
                    input["B"][j].save(name2)
                    k+=1






