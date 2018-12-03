from __future__ import division
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset,get_folders_name
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os


class LesionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self,opt):
        self.opt = opt
        self.dataset_mode = True
        self.root = "/media/ahmed/DiskA1/lesion_dataset2/"
        phase = opt.phase
        self.dir_A = os.path.join(self.root, phase + 'A')
        self.dir_B = os.path.join(self.root, phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)

        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        transform_ = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_)



    def __getitem__(self, index):


        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        if(self.opt.mode == "unaligned"):
            B_path = self.B_paths[random.randint(0,  self.B_size - 1)]


        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B,'A_paths': A_path,'  B_paths': B_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'LesionDataset'