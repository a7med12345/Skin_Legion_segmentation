import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms

class LesionModel(BaseModel):
    def name(self):
        return 'LesionModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(dataset_mode='lesion')
        parser.set_defaults(netG='unet_512')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A','fake_B','real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.opt = opt

        if self.isTrain:

            # define loss functions
            self.criterion = torch.nn.MSELoss()
            if(self.opt.loss_type == "bce"):
                self.criterion = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                         lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)



    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'B'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):

        self.fake_B = self.netG(self.real_A)


    def backward(self):

        self.loss_G = self.criterion( self.fake_B, self.real_B)

        self.loss_G.backward()



    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

