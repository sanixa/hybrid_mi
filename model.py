import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import copy
import torchvision.models as models

############################################################################################################################
# ###########################################################################################################################
# #################################################CNN##############################################################
# ###########################################################################################################################
# ###########################################################################################################################

class ResNetResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel=3, downsampling=1, conv_shortcut=False, TS=False, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.kernel, self.downsampling = kernel, downsampling
        self.conv_shortcut= conv_shortcut
        if TS:
            self.activate = TemperedSigmoid()
        else:
            self.activate = nn.ReLU()
        self.shortcut = nn.Conv2d(self.in_channels, filters *4, kernel_size=1,
                      stride=self.downsampling) if self.conv_shortcut else nn.MaxPool2d(kernel_size=1, stride=self.downsampling)

        self.BN_1 = nn.BatchNorm2d(self.in_channels, eps=1.001e-5)
        self.Conv_1 = nn.Conv2d(self.in_channels, filters, kernel_size=1, stride=1, bias=False)
        self.BN_2 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.zeroPad_1 = nn.ZeroPad2d((1,1,1,1))
        self.Conv_2 = nn.Conv2d(filters, filters, kernel_size=self.kernel, stride=self.downsampling, bias=False)
        self.BN_3 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.Conv_3 = nn.Conv2d(filters , filters *4, kernel_size=1)
        
    def forward(self, x):
        x = self.BN_1(x)
        x = self.activate(x)

        residual = self.shortcut(x)

        x = self.Conv_1(x)
        x = self.BN_2(x)
        x = self.activate(x)
        x = self.zeroPad_1(x)
        x = self.Conv_2(x)
        x = self.BN_3(x)
        x = self.activate(x)
        x = self.Conv_3(x)
        x += residual
        return x

class ResNet18v2_cifar10(nn.Module):
    def __init__(self, classes=10, *args, **kwargs):
        super().__init__()
        self.classes = classes
        
        '''
        self.zeroPad_1 = nn.ZeroPad2d((3,3,3,3))
        self.Conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False)
        self.zeroPad_2 = nn.ZeroPad2d((1,1,1,1))
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        ##----block
        self.block_1 = ResNetResidualBlock(64, 16, conv_shortcut=True)
        self.block_2 = ResNetResidualBlock(64, 16, downsampling=2)
        self.block_3 = ResNetResidualBlock(64, 32, conv_shortcut=True)
        self.block_4 = ResNetResidualBlock(128, 32, downsampling=2)
        self.block_5 = ResNetResidualBlock(128, 64, conv_shortcut=True)
        self.block_6 = ResNetResidualBlock(256, 64, downsampling=2)
        self.block_7 = ResNetResidualBlock(256, 128, conv_shortcut=True)
        self.block_8 = ResNetResidualBlock(512, 128)
        
        self.BN_1 = nn.BatchNorm2d(512, eps=1.001e-5)
        self.activate = nn.ReLU()
        self.GAP_1 = nn.AvgPool2d(1,1)
        self.fc_1 = nn.Linear(1,classes)
        '''
        self.model = nn.Sequential(
            nn.ZeroPad2d((3,3,3,3)),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.ZeroPad2d((1,1,1,1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResNetResidualBlock(64, 16, conv_shortcut=True),
            ResNetResidualBlock(64, 16, downsampling=2),
            ResNetResidualBlock(64, 32, conv_shortcut=True),
            ResNetResidualBlock(128, 32, downsampling=2),
            ResNetResidualBlock(128, 64, conv_shortcut=True),
            ResNetResidualBlock(256, 64, downsampling=2),
            ResNetResidualBlock(256, 128, conv_shortcut=True),
            ResNetResidualBlock(512, 128),
            nn.BatchNorm2d(512, eps=1.001e-5),
            nn.ReLU(),
            nn.AvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,classes))
        #self.GAP_1 = nn.AvgPool2d(1)
        #self.fc_1 = nn.Linear(1,classes)
        #self.flatten = nn.Flatten()
    def forward(self, x):
        '''
        x = self.zeroPad_1(x)
        x = self.Conv_1(x)
        x = self.zeroPad_2(x)
        x = self.maxpool_1(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)

        x = self.BN_1(x)
        x = self.activate(x)
        '''
        x = self.model(x)
        #x = self.GAP_1(x)
        #x = self.flatten(x)
        #x = self.fc_1(x)
        return x


############################################################################################################################
# ###########################################################################################################################
# #################################################GAN##############################################################
# ###########################################################################################################################
# ###########################################################################################################################


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)



def pixel_norm(x, eps=1e-10):
    '''
    Pixel normalization
    :param x:
    :param eps:
    :return:
    '''
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdim=True) + eps)

#@torchsnooper.snoop()
class GeneratorDCGAN_cifar(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Tanh()):
        super(GeneratorDCGAN_cifar, self).__init__()

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, z_dim * 1 * 1)
        deconv1 = nn.ConvTranspose2d(z_dim, model_dim * 4, 4, 1, 0, bias=False)
        deconv2 = nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False)
        deconv3 = nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False)
        deconv4 = nn.ConvTranspose2d(model_dim, 3, 4, 2, 1, bias=False)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.deconv4 = deconv4
        self.BN_1 = nn.BatchNorm2d(model_dim * 4)
        self.BN_2 = nn.BatchNorm2d(model_dim * 2)
        self.BN_3 = nn.BatchNorm2d(model_dim)
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact

        ''' reference by https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
        nn.ConvTranspose2d(z_dim, model_dim * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(model_dim * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(model_dim * 8, model_dim * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(model_dim, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
        '''

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, self.z_dim, 1, 1)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = self.BN_1(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.BN_2(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.BN_3(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv4(output)
        output = self.outact(output)

        return output.view(-1, 3 * 32 * 32)


nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class Generator_celeba(nn.Module):
    def __init__(self, ngpu):
        super(Generator_celeba, self).__init__()
        self.ngpu = ngpu
        self.num_classes = 2
        self.z_dim = 100

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def cnn_celeba():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    return model_ft
