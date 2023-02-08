import torch
from torch.nn.parameter import Parameter
from dim_wgan.Unet import *
from dim_wgan.utils import *

def initialization(model):
    for i in model.modules():
        if not isinstance(i, nn.modules.container.Sequential):
            classname = i.__class__.__name__
            if hasattr(i, "weight"):
                if classname.find("Conv") != -1:
                    nn.init.normal_(i.weight)
                elif classname.find("BatchNorm2d") != -1:
                    nn.init.normal_(i.weight.data, 1.0, 0.02)
            if hasattr(i, 'bias'):
                nn.init.zeros_(i.bias)
                if classname.find("BatchNorm2d") != -1:
                    nn.init.constant_(i.bias.data, 0.0)

class Vectorize(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

#TODO: to check how/why it works and fix it!
class MinibatchDiscrimination1d(nn.Module):
    def __init__(self, in_features, out_features, intermediate_features=16):
        super(MinibatchDiscrimination1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features

        self.T = Parameter(
            torch.Tensor(in_features, out_features, intermediate_features)
        )
        nn.init.normal_(self.T)

    def forward(self, x):
        r"""Computes the output of the Minibatch Discrimination Layer

        Args:
            x (torch.Tensor): A Torch Tensor of dimensions :math: `(N, infeatures)`

        Returns:
            3D Torch Tensor of size :math: `(N,infeatures + outfeatures)` after applying Minibatch Discrimination
        """
        M = torch.mm(x, self.T.view(self.in_features, -1))
        M = M.view(-1, self.out_features, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        # Broadcasting reduces the matrix subtraction to the form desired in the paper
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        return torch.cat([x, out], 1)

##############
# Generators #
##############

class GeneratorInline(nn.Module):
    def __init__(self, hps):
        super(GeneratorInline, self).__init__()
        self.generator_layer_0 = nn.Sequential(
            # input: 1X1X249 channels
            nn.ConvTranspose1d(100, hps['nchan'] * 4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 4)
        )
        self.generator_layer_1 = nn.Sequential(
            # intermediate: 2X2X[64*8]
            nn.ConvTranspose1d(hps['nchan'] * 4, hps['nchan'] * 8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 8)
        )
        self.generator_layer_2 = nn.Sequential(
            # intermediate: 4X4X[64*4]
            nn.ConvTranspose1d(hps['nchan'] * 8, hps['nchan'] * 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 16)
        )
        self.generator_layer_3 = nn.Sequential(
            # intermediate: 4X4X[64*4]
            nn.ConvTranspose1d(hps['nchan'] * 16, 1, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(1)
        )
        self.generator_layer_4 = nn.Sequential(
            # intermediate: 8X8X[64*8]
            # output: 1X16
            Vectorize(),
            nn.Linear(17, 248)
        )
        initialization(self.generator_layer_3)
        initialization(self.generator_layer_2)
        initialization(self.generator_layer_1)
        initialization(self.generator_layer_0)

    def forward(self, x):
        #print(x.shape)
        i = 0
        outputs = []
        # print('input: ', x.size())
        layer_name = lambda x: f"generator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            x = layer(x)
            outputs.append(x)
            #print(f'{x.shape}')
            i += 1
        return torch.unsqueeze(outputs[-1], dim=-1)

class Generator(nn.Module):
    def __init__(self, hps, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.generator_layer_0 = nn.Sequential(
            # input is Z, going into the convolution (1,100,1)
            nn.ConvTranspose1d(100, hps['nchan'] * 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 8),
            nn.ReLU(True)
        )
        self.generator_layer_1 = nn.Sequential(
            # state size. (ngf*8) x 3 x 3
            nn.ConvTranspose1d(hps['nchan'] * 8, hps['nchan'] * 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 4),
            nn.ReLU(True)
        )
        self.generator_layer_2 = nn.Sequential(
            # state size. (ngf*4) x 6 x 6
            nn.ConvTranspose1d(hps['nchan'] * 4, hps['nchan'] * 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 2),
            nn.ReLU(True)
        )
        self.generator_layer_3 = nn.Sequential(
            # state size. (ngf) x 12 x 12
            nn.ConvTranspose1d(hps['nchan'] * 2, hps['nc'], kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2)
            # state size. (nc) x 13 x 13
        )
        initialization(self.generator_layer_0)
        initialization(self.generator_layer_1)
        initialization(self.generator_layer_2)
        initialization(self.generator_layer_3)

    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.size())
        layer_name = lambda x: f"generator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            x = layer(x)
            outputs.append(x)
            #print(f'{x.shape}')
            i+=1
        return outputs[-1]

class ResNetGenerator(nn.Module):
    def __init__(self, hps, ngpu=1):
        super(ResNetGenerator, self).__init__()
        self.ngpu = ngpu
        self.generator_layer_0 = nn.Sequential(
            # input is Z, going into the convolution (batch_size,100,1)
            nn.ConvTranspose1d(100, hps['nchan'] * 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 8),
            nn.ReLU()
        )
        self.residual_layer_0 = nn.Sequential(
            # state size. batch_size x (ngf*8) x 3
            nn.ConvTranspose1d(hps['nchan'] * 8, hps['nchan'] * 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 16),
            nn.ReLU(True),
            nn.ConvTranspose1d(hps['nchan'] * 16, hps['nchan'] * 8, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm1d(hps['nchan'] * 8),
            nn.ReLU()
        )
        self.generator_layer_1 = nn.Sequential(
            # state size. batch_size x (ngf*8) x 3
            nn.ConvTranspose1d(hps['nchan'] * 8, hps['nchan'] * 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 4),
            nn.ReLU()
        )
        self.residual_layer_1 = nn.Sequential(
            # state size. batch_size x (ngf*4) x 6
            nn.ConvTranspose1d(hps['nchan'] * 4, hps['nchan'] * 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(hps['nchan'] * 8, hps['nchan'] * 4, kernel_size=3, stride=1, padding=4),
            nn.BatchNorm1d(hps['nchan'] * 4),
            nn.ReLU()
        )
        self.generator_layer_2 = nn.Sequential(
            # state size. batch_size x (ngf*4) x 6
            nn.ConvTranspose1d(hps['nchan'] * 4, hps['nchan'] * 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(hps['nchan'] * 2),
            nn.ReLU()
        )
        self.generator_layer_3 = nn.Sequential(
            # state size. batch_size x (ngf*4) x 12
            nn.ConvTranspose1d(hps['nchan'] * 2, hps['nc'], kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2)
            # state size. batch_size x (nc) x 13
        )

        initialization(self.generator_layer_0)
        initialization(self.residual_layer_0)
        initialization(self.generator_layer_1)
        initialization(self.residual_layer_1)
        initialization(self.generator_layer_2)
        initialization(self.generator_layer_3)

    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.size())
        generator_name = lambda x: f"generator_layer_{x}"
        residual_name = lambda x: f"residual_layer_{x}"
        while hasattr(self, generator_name(i)):
            layer = getattr(self, generator_name(i))
            x = layer(x)
            #print(f'{layer}, {x.shape}')
            if i < 2:
                x_in = x
                layer = getattr(self, residual_name(i))
                x = layer(x)
                #print(f'{layer}, {x.shape}')
                x += x_in
            outputs.append(x)
            i+=1
        return outputs[-1]


class Unet_Generator(nn.Module):
    def __init__(self, hps):
        super(Unet_Generator, self).__init__()
        self.ngpu = hps['ngpu']
        self.main = nn.Sequential(
            Unet1d(hps),
            nn.LeakyReLU(0.2)
        )
        initialization(self.main)

    def forward(self, x):
        return self.main(x)


############################
# Discriminators - Critics #
############################

class Discriminator(nn.Module):
    def __init__(self, hps, wgan=False, etest=True):
        super(Discriminator, self).__init__()
        self.ngpu = hps['ngpu']
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc)
            nn.Conv1d(hps['nc'], hps['ndf'], kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf)
            nn.Conv1d(hps['ndf'], hps['ndf'] * 2, kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.BatchNorm1d(hps['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 2, hps['ndf'] * 4, kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.BatchNorm1d(hps['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 4, hps['nc'], 2 if etest else 1, stride=1, padding=0)
        )
        # For WGAN DON'T Apply Sigmoid. It should not be a probability any more!!!
        if not wgan:
            self.discriminator_layer_4 = nn.Sequential(
                nn.Sigmoid()
            )

    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.shape)
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            #print(f'{x.shape}, {layer}')
            x = layer(x)
            #print(f'{x[0]}')
            #print(f'{x.shape}')
            outputs.append(x)
            i+=1
        return outputs

class SimpleDiscriminator(nn.Module):
    def __init__(self, hps, wgan=False, etest=True):
        super(SimpleDiscriminator, self).__init__()
        self.ngpu = hps['ngpu']
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc)
            nn.Conv1d(hps['nc'], hps['ndf'], kernel_size=3 if etest else 1, stride=2 if etest else 1,
                      padding=1 if etest else 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf)
            nn.Conv1d(hps['ndf'], hps['ndf'] * 2, kernel_size=3 if etest else 1, stride=2 if etest else 1,
                      padding=1 if etest else 0),
            nn.BatchNorm1d(hps['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 2, hps['ndf'] * 4, kernel_size=3 if etest else 1, stride=2 if etest else 1,
                      padding=1 if etest else 0),
            nn.BatchNorm1d(hps['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 4, 1, 2 if etest else 1, stride=1, padding=0)
        )
        # For WGAN DON'T Apply Sigmoid. It should not be a probability any more!!!
        if not wgan:
            self.discriminator_layer_4 = nn.Sequential(
                nn.Sigmoid()
            )

    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.shape)
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            #print(f'{x.shape}, {layer}')
            x = layer(x)
            #print(f'{x[0]}')
            #print(f'{x.shape}')
            outputs.append(x)
            i+=1
        return outputs


class InstNDiscriminator(nn.Module):
    def __init__(self, hps, wgan=False, etest=True):
        super(InstNDiscriminator, self).__init__()
        self.ngpu = hps['ngpu']
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc)
            nn.Conv1d(hps['nc'], hps['ndf'], kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf)
            nn.Conv1d(hps['ndf'], hps['ndf'] * 2, kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.InstanceNorm1d(hps['ndf'] * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 2, hps['ndf'] * 4, kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.InstanceNorm1d(hps['ndf'] * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 4, 1, 2 if etest else 1, stride=1, padding=0)
        )
        # For WGAN DON'T Apply Sigmoid. It should not be a probability any more!!!
        if not wgan:
            self.discriminator_layer_4 = nn.Sequential(
                nn.Sigmoid()
            )


    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.shape)
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            #print(f'{x.shape}, {layer}')
            x = layer(x)
            #print(f'{x[0]}')
            #print(f'{x.shape}')
            outputs.append(x)
            i+=1
        return outputs


class LayerNDiscriminator(nn.Module):
    def __init__(self, hps, wgan=False, etest=True):
        super(LayerNDiscriminator, self).__init__()
        self.ngpu = hps['ngpu']
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc)
            nn.Conv1d(hps['nc'], hps['ndf'], kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf)
            nn.Conv1d(hps['ndf'], hps['ndf'] * 2, kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.LayerNorm(hps['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 2, hps['ndf'] * 4, kernel_size=3 if etest else 1, stride=2 if etest else 1, padding=1 if etest else 0),
            nn.LayerNorm(hps['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*2)
            nn.Conv1d(hps['ndf'] * 4, 1, 2 if etest else 1, stride=1, padding=0)
        )
        # For WGAN DON'T Apply Sigmoid. It should not be a probability any more!!!
        if not wgan:
            self.discriminator_layer_4 = nn.Sequential(
                nn.Sigmoid()
            )


    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.shape)
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            #print(f'{x.shape}, {layer}')
            x = layer(x)
            #print(f'{x[0]}')
            #print(f'{x.shape}')
            outputs.append(x)
            i+=1
        return outputs
