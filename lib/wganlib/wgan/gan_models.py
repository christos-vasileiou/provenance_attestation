import torch
from torch.nn.parameter import Parameter
from wgan.Unet import *

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

class Vectorize(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class VectorizedLinear(nn.Module):
    def __init__(self, osize):
        self.vectorize = Vectorize()
        self.linear = lambda x: nn.Linear(x, osize)
        self.osize = osize
    def forward(self, x):
        x = self.vectorize(x)
        x = self.linear(x)
        return x

class GeneratorInline(nn.Module):
    def __init__(self, hps):
        super(GeneratorInline, self).__init__()
        seed = hps['seed'] if 'seed' in hps.keys() else 100
        nchan = hps['nchan'] if 'nchan' in hps.keys() else 64
        osize = hps['osize'] if 'osize' in hps.keys() else 248

        self.generator_layer_0 = nn.Sequential(
            # input: [batch_size, 50, 1]
            nn.ConvTranspose1d(seed, nchan * 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(nchan * 4)
        )
        self.generator_layer_1 = nn.Sequential(
            # intermediate: [batch_size, 256, 3]
            nn.ConvTranspose1d(nchan * 4, nchan * 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(nchan * 8)
        )
        self.generator_layer_2 = nn.Sequential(
            # intermediate: [batch_size, 512, 5]
            nn.ConvTranspose1d(nchan * 8, nchan * 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(nchan * 16)
        )
        self.generator_layer_3 = nn.Sequential(
            # intermediate: [batch_size, 1024, 7]
            nn.ConvTranspose1d(nchan * 16, 1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(1)
        )
        self.generator_layer_4 = nn.Sequential(
            # intermediate: [batch_size, 1, 9]
            # Vectorize: batch_sizeX9
            Vectorize(),
            nn.Linear(9, osize)
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
            #print(f'{x.shape}')
            x = layer(x)
            outputs.append(x)
            #print(f'{x.shape}')
            i += 1
        return torch.unsqueeze(outputs[-1], dim=-1)


class GeneratorInlineLinear(nn.Module):
    def __init__(self, hps):
        super(GeneratorInline, self).__init__()
        seed = hps['seed'] if 'seed' in hps.keys() else 100
        nchan = hps['nchan'] if 'nchan' in hps.keys() else 64
        osize = hps['osize'] if 'osize' in hps.keys() else 248

        self.generator_layer_0 = nn.Sequential(
            # input: [batch_size, 50, 1]
            nn.Linear(seed, nchan * 4),
            nn.Dropout(0.2)
        )
        self.generator_layer_1 = nn.Sequential(
            # intermediate: [batch_size, 256, 3]
            nn.Linear(nchan * 4, nchan * 8),
            nn.Dropout(0.2)
        )
        self.generator_layer_2 = nn.Sequential(
            # intermediate: [batch_size, 512, 5]
            nn.Linear(nchan * 8, nchan * 16),
            nn.Dropout(0.2)
        )
        self.generator_layer_3 = nn.Sequential(
            # intermediate: [batch_size, 1024, 7]
            nn.Linear(nchan * 16, nchan * 32),
            nn.Dropout(0.2)
        )
        self.generator_layer_4 = nn.Sequential(
            # intermediate: [batch_size, 1, 9]
            # Vectorize: batch_sizeX9
            nn.Linear(nchan * 32, osize)
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
            #print(f'{x.shape}')
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

class DiscriminatorInline(nn.Module):
    def __init__(self, hps, wgan=False, etest=True):
        super(DiscriminatorInline, self).__init__()
        i=0
        ndf = hps['ndf'] if 'ndf' in hps.keys() else 64
        osize = hps['osize'] if 'osize' in hps.keys() else 248
        layer_name = lambda x: f"discriminator_layer_{x}"
        #print(osize)
        while osize > 1:
            ks = 3 if i==0 else 1
            s = 2
            pad = 0
            #pad = 1 if osize == 2 else pad
            setattr(self, layer_name(i), nn.Sequential(
                nn.Conv1d(1 if i==0 else (i)*ndf, (i+1)*ndf, kernel_size=ks, stride=s, padding=pad),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            osize = (osize + 2*pad - (ks-1)-1)/s+1
            if osize%(osize//1)!=0:
                osize = osize//1
            #osize = osize if osize != 3 else 1
            i+=1
            #print(osize)
        # For WGAN DON'T Apply Sigmoid. It should not be a probability any more!!!
        if not wgan:
            setattr(self, layer_name(i), nn.Sequential(
                nn.Sigmoid()
            ))
            i+=1
    def forward(self, x):
        i=0
        outputs = []
        #print('input: ', x.shape)
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            #print(f'{x.shape}')
            x = layer(x)
            #print(f'{x.shape}')
            outputs.append(x)
            i+=1
        return outputs


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

class MultiDimDiscriminator(nn.Module):
    def __init__(self, hps, wgan=False, etest=True):
        super(MultiDimDiscriminator, self).__init__()
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
            #nn.Conv1d(hps['ndf'] * 4, 1, 2 if etest else 1, stride=1, padding=0)
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
            nn.Conv1d(hps['ndf'] * 4, hps['nc'], 2 if etest else 1, stride=1, padding=0)
            #nn.Conv1d(hps['ndf'] * 4, 1, 2 if etest else 1, stride=1, padding=0)
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
