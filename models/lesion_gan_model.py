import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms

class LesionGanModel(BaseModel):
    def name(self):
        return 'LesionGanModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(dataset_mode='lesion')
        parser.set_defaults(netG='unet_512')
        parser.set_defaults(netD='basic')

        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G','D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A','fake_B','real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['G','D']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = False
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                          self.gpu_ids)            # define loss functions
            self.criterion = torch.nn.MSELoss()
            #self.criterion = torch.nn.BCEWithLogitsLoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                         lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)



    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'B'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):

        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):

        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L2 = self.criterion(self.fake_B, self.real_B) * 100

        self.loss_G = self.loss_G_GAN + self.loss_G_L2
        self.loss_G.backward()



    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

