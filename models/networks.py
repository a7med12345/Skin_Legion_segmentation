import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from collections import namedtuple
from torchvision import models
import torch.nn.functional as F

###############################################################################
# Helper Functions & classes
###############################################################################

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    self.relu = nn.ReLU()
  def forward(self, x):
    out = self.conv(x)
    out = self.relu(out)

    out = torch.cat((x, out), 1)
    return out



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "test":
        net = Unet(input_nc,output_nc)
    elif netG == "test2":
        net = Unet2(input_nc, output_nc)
    elif netG == "wespe":
        net = Generator_wespe()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'wespe':
        net = Discriminator_wespe()
    elif netD == 'test':
        net = Discriminator_unet()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)


    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan= False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Content(torch.nn.Module):
    def __init__(self, pretrained_name="resnet18"):
        super(Content, self).__init__()
        self.model_keys, self.model = self.get_pretrained_model_info(pretrained_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.trunc_model = nn.Sequential(*list(self.model.children())[:-3])


    def get_pretrained_model_info(self,pretrained_name):

            # upload the pretrained model
            x = "models." + pretrained_name + "(pretrained=True)"
            # exec("model="+x)
            model = eval(x)
            # print(model)
            # Get keys and values of the model
            list_keys = []
            list_value = []
            for key in model.state_dict():
                value = model.state_dict().get(key)
                list_keys.append(key)
                list_value.append(value)

            l = []
            for key in list_keys:
                a = key.split('.')
                for i in range(0, len(a) - 2):
                    a[i] = a[i] + '.'

                s = ''.join(a[:-1])
                if s not in l:
                    l.append(s)

            return l,model

    def forward(self, X):
        return self.trunc_model(X)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def __call__(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf*2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:

            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:

            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:


            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print('input',x.shape)
        #return self.model(x)
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)




#Super Resolution networks







# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)




























#########Wespe
class ConvBlock_wespe(nn.Module):

    def __init__(self):
        super(ConvBlock_wespe, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        y = self.relu(self.instance_norm1(self.conv1(x)))
        y = self.relu(self.instance_norm2(self.conv2(y))) + x
        return y


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class GrayLayer(nn.Module):

    def __init__(self):
        super(GrayLayer, self).__init__()

    def forward(self, x):
        result = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return result.unsqueeze(1)


class Generator_wespe(nn.Module):

    def __init__(self):
        super(Generator_wespe, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.blocks = nn.Sequential(
            ConvBlock_wespe(),
            ConvBlock_wespe(),
            ConvBlock_wespe(),
            ConvBlock_wespe(),
        )
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, 9, padding=4)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        x = self.act(self.conv1(x))
        x = self.blocks(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))


        return x


class Discriminator_wespe(nn.Module):

    def __init__(self, input_ch=3):
        super(Discriminator_wespe, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_ch, 48, 11, stride=4, padding=5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(48, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
        )

        self.fc = nn.Linear(128*8*8, 1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128*8*8)
        x = self.fc(x)
        x = self.out(x)

        return x

class GrayLayer(nn.Module):

    def __init__(self):
        super(GrayLayer, self).__init__()

    def forward(self, x):
        result = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return result.unsqueeze(1)


###DenseNet
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out




#Attention Network
#Attention Network
class AttentionLayer(nn.Module):
    def __init__(self, nChannels):
        super(AttentionLayer, self).__init__()

        self.conv1 = nn.Conv2d(nChannels, nChannels*2, kernel_size=4,padding=1, stride=2,bias=True)
        self.deconv = nn.ConvTranspose2d(nChannels*2, nChannels, kernel_size=4,padding=1, stride=2,bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.deconv(out))
        #out = out.view(out.size(0), -1)
        out = F.softmax(out,dim=1)
        return out*x


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel,  kernel_size=4, stride=2,padding=1,bias=True)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.LeakyReLU()
        self.batch_normalization = batch_normalization

    def forward(self,x):
        input =x
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)

        return x,input

class DeConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(DeConvBlock, self).__init__()

        self.conv1 = torch.nn.ConvTranspose2d(input_channel, output_channel,  kernel_size=4, stride=2,padding=1,bias=True)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)

        self.relu = torch.nn.Tanh()
        self.batch_normalization = batch_normalization

    def forward(self,x):
        input = x
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)

        return x,input


class Unet(torch.nn.Module):
    def __init__(self,input_channel,output_channel,num_blocks=8,apply_attention=False):
        super(Unet, self).__init__()
        self.num_block = num_blocks
        self.apply_attention = apply_attention

        #conv Blocks
        self.convblock1 = ConvBlock(input_channel,64)
        self.convblock2 = ConvBlock(64,128)
        self.convblock3 = ConvBlock(128,256)
        self.convblock4 = ConvBlock(256, 512)
        self.convblock5 = ConvBlock(512, 1024)

        #deconv Blocks
        self.deconvblock1 = DeConvBlock(1024, 512)
        self.deconvblock2 = DeConvBlock(1024, 256)
        self.deconvblock3 = DeConvBlock(512, 128)
        self.deconvblock4 = DeConvBlock(256, 64)
        self.deconvblock5 = DeConvBlock(128, 3)

        #Attention networks
        self.att1 = AttentionLayer(1024)
        self.att2 = AttentionLayer(1024)
        self.att3 = AttentionLayer(512)
        self.att4 = AttentionLayer(256)
        self.att5 = AttentionLayer(128)

    def set_gradient(self,bool=False):
        for param in self.convblock1.parameters():
            param.requires_grad = bool
        for param in self.convblock2.parameters():
             param.requires_grad = bool
        for param in self.convblock3.parameters():
            param.requires_grad = bool
        for param in self.convblock4.parameters():
            param.requires_grad = bool
        for param in self.convblock5.parameters():
            param.requires_grad = bool

        for param in self.deconvblock1.parameters():
            param.requires_grad = bool
        for param in self.deconvblock2.parameters():
            param.requires_grad = bool
        for param in self.deconvblock3.parameters():
            param.requires_grad = bool
        for param in self.deconvblock4.parameters():
            param.requires_grad = bool
        for param in self.deconvblock5.parameters():
            param.requires_grad = bool

    def forward(self,x):

        x, input1 = self.convblock1(x)
        x, input2 = self.convblock2(x)
        x, input3 = self.convblock3(x)
        x, input4 = self.convblock4(x)
        x, input5 = self.convblock5(x)

        if(self.apply_attention):
            x =self.att1(x)

        x,input6 = self.deconvblock1(x)
        x_cat_1 = torch.cat([input5,x], 1)

        if(self.apply_attention):
            x_cat_1 = self.att2(x_cat_1)

        x, input7 = self.deconvblock2(x_cat_1)
        x_cat_2 = torch.cat([input4,x], 1)

        if(self.apply_attention):
            x_cat_2 = self.att3(x_cat_2)

        x,input8 = self.deconvblock3(x_cat_2)
        x_cat_3 = torch.cat([input3,x], 1)

        if(self.apply_attention):
            x_cat_3 = self.att4(x_cat_3)


        x, input9 = self.deconvblock4(x_cat_3)
        x_cat_4 = torch.cat([input2,x], 1)

        if(self.apply_attention):
            x_cat_4 = self.att5(x_cat_4)

        x, input10 = self.deconvblock5(x_cat_4)

        return x


class Unet2(torch.nn.Module):
    def __init__(self,input_channel,output_channel,num_blocks=8,apply_attention=False):
        super(Unet2, self).__init__()
        self.num_block = num_blocks
        self.apply_attention = apply_attention

        #conv Blocks
        self.convblock1 = ConvBlock(input_channel,64)
        self.convblock2 = ConvBlock(64,128)
        self.convblock3 = ConvBlock(128,256)
        self.convblock4 = ConvBlock(256, 512)
        self.convblock5 = ConvBlock(512, 1024)

        #deconv Blocks
        self.deconvblock1 = DeConvBlock(1024, 512)
        self.deconvblock2 = DeConvBlock(512, 256)
        self.deconvblock3 = DeConvBlock(256, 128)
        self.deconvblock4 = DeConvBlock(128, 64)
        self.deconvblock5 = DeConvBlock(64, 3)

        #Attention networks
        self.att1 = AttentionLayer(1024)
        self.att2 = AttentionLayer(512)
        self.att3 = AttentionLayer(256)
        self.att4 = AttentionLayer(128)
        self.att5 = AttentionLayer(64)

    def set_gradient(self,bool=False):
        for param in self.convblock1.parameters():
            param.requires_grad = bool
        for param in self.convblock2.parameters():
             param.requires_grad = bool
        for param in self.convblock3.parameters():
            param.requires_grad = bool
        for param in self.convblock4.parameters():
            param.requires_grad = bool
        for param in self.convblock5.parameters():
            param.requires_grad = bool

        for param in self.deconvblock1.parameters():
            param.requires_grad = bool
        for param in self.deconvblock2.parameters():
            param.requires_grad = bool
        for param in self.deconvblock3.parameters():
            param.requires_grad = bool
        for param in self.deconvblock4.parameters():
            param.requires_grad = bool
        for param in self.deconvblock5.parameters():
            param.requires_grad = bool

    def forward(self,x):

        x, input1 = self.convblock1(x)
        x, input2 = self.convblock2(x)
        x, input3 = self.convblock3(x)
        x, input4 = self.convblock4(x)
        x, input5 = self.convblock5(x)

        if(self.apply_attention):
            x =self.att1(x)
        x,input6 = self.deconvblock1(x)
        x_cat_1 = x + input5

        if(self.apply_attention):
            x_cat_1 = self.att2(x_cat_1)

        x, input7 = self.deconvblock2(x_cat_1)
        x_cat_2 = x +input4

        if(self.apply_attention):
            x_cat_2 = self.att3(x_cat_2)

        x,input8 = self.deconvblock3(x_cat_2)
        x_cat_3 = x + input3

        if(self.apply_attention):
            x_cat_3 = self.att4(x_cat_3)

        x, input9 = self.deconvblock4(x_cat_3)
        x_cat_4 = x + input2

        if(self.apply_attention):
            x_cat_4 = self.att5(x_cat_4)

        x, input10 = self.deconvblock5(x_cat_4)

        return x


class Discriminator_unet(torch.nn.Module):
    def __init__(self,input_channel,output_channel,num_blocks=8,apply_attention=False):
        super(Unet, self).__init__()
        self.num_block = num_blocks
        self.apply_attention = apply_attention

        #conv Blocks
        self.convblock1 = ConvBlock(input_channel,64)
        self.convblock2 = ConvBlock(64,128)
        self.convblock3 = ConvBlock(128,256)
        self.convblock4 = ConvBlock(256, 512)
        self.convblock5 = ConvBlock(512, 1024)

        #deconv Blocks
        self.deconvblock1 = DeConvBlock(1024, 512)
        self.deconvblock2 = DeConvBlock(1024, 256)
        self.deconvblock3 = DeConvBlock(512, 128)
        self.deconvblock4 = DeConvBlock(256, 64)
        self.deconvblock5 = DeConvBlock(128, 3)

        #Attention networks
        self.att1 = AttentionLayer(1024)
        self.att2 = AttentionLayer(1024)
        self.att3 = AttentionLayer(512)
        self.att4 = AttentionLayer(256)


    def forward(self,x):

        x, input1 = self.convblock1(x)
        x = x + input1
        if (self.apply_attention):
            x = self.att1(x)
        x, input2 = self.convblock2(x)
        x = x + input2
        if (self.apply_attention):
            x = self.att2(x)
        x, input3 = self.convblock3(x)
        x = x + input3
        if (self.apply_attention):
            x = self.att3(x)
        x, input4 = self.convblock4(x)
        x = x + input4
        if (self.apply_attention):
            x = self.att4(x)
        x, input5 = self.convblock5(x)
        x = x + input5


        return x

class Discriminator_unet(torch.nn.Module):
    def __init__(self, input_channel=3):
        super(Discriminator_unet, self).__init__()

        # conv Blocks
        self.convblock1 = ConvBlock1(input_channel, 64,3)
        self.convblock2 = ConvBlock1(64, 128,5)
        self.convblock3 = ConvBlock1(128, 256,7)
        self.convblock4 = ConvBlock1(256, 512,9)
        self.convblock5 = ConvBlock1(512, 1024,9)




        self.fc = nn.Linear(1024 * 9 * 9, 1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, x):

        x, input1 = self.convblock1(x)

        x, input2 = self.convblock2(x)

        x, input3 = self.convblock3(x)

        x, input4 = self.convblock4(x)

        x, input5 = self.convblock5(x)

        x = x.view(-1, 1024 * 9 * 9)
        x = self.fc(x)
        x = self.out(x)
        return x

class ConvBlock1(torch.nn.Module):
    def __init__(self, input_channel, output_channel,kernel_size, batch_normalization=True):
        super(ConvBlock1, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel,  kernel_size=kernel_size, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self,x):
        input =x
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)

        return x,input