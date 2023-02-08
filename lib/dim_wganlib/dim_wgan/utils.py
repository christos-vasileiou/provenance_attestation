from dim_wgan.gan_models import *
import torch.nn as nn
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

def save_model(maskset, training_typo, netG):
    path = f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Generator" + str(training_typo) + ".pt"
    package_name = f"{maskset}msks-Generator" + str(training_typo)
    resource_name = "model.pkl"
    with torch.package.PackageExporter(path) as exp:
        exp.extern("numpy.**")
        exp.intern("models.**")
        exp.save_pickle(package_name, resource_name, netG)

def load_model(maskset, training_typo):
    path = f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Generator" + str(training_typo) + ".pt"
    package_name = f"{maskset}msks-Generator" + str(training_typo)
    imp = torch.package.PackageImporter(path)
    loaded_model = imp.load_pickle(package_name, resource_name="model.pkl")
    return loaded_model

def calculate_gradient_penalty(real, fake, netD):
    """
    :param real: real data
    :param fake: fake produced data
    :param netD: named as critic
    :param lambda_term: lambda given by default. You can tune it!
    :return: gradient penalty to smooth Lipschitz contraint.
    """
    lambda_term = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real = real.to(device)
    fake = fake.to(device)
    
    eta = torch.FloatTensor(real.size(0), 1, 1).uniform_(0, 1).repeat(1, real.size(1), real.size(2))
    eta = eta.to(device)
    interpolated = eta * real + ((1 - eta) * fake)

    interpolated = interpolated.to(device)
    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)[-1]

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def convert_size(size):
    if size < 1024:
        return size
    elif (size >= 1024) and (size < (1024 * 1024)):
        return "%.2f KB" % (size / 1024)
    elif (size >= (1024 * 1024)) and (size < (1024 * 1024 * 1024)):
        return "%.2f MB" % (size / (1024 * 1024))
    else:
        return "%.2f GB" % (size / (1024 * 1024 * 1024))


def is_unique(u):
    # a = s.to_numpy() # s.values (pandas<0.24)
    return (u[0] == u).all()


def generate_sample(G, noise, device, dtype):
    G.to(device)
    noise = noise.to(device)
    # noise = torch.randn(1, dataset.shape[1], dataset.shape[2], device=device)
    # noise = torch.randn(1, 100, 1, device=device)

    output = G(noise).to(device)
    return output.cpu().detach().numpy()


def plotResults(results):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    ax1.plot(results.index, results["Train Loss"],
             results.index, results["Test Loss"],
             results.index, results["Validation Loss"],
             results.index, results["Min Validation Loss"])
    ax1.legend(["Train", "Test", "Validation", "Min Validation"])
    ax1.set_title("Loss")

    ax2.plot(results.index, results["Validation Loss"] / results["Min Validation Loss"] - 1)
    ax2.set_title("Val Loss/Min Val Loss - 1")

    plt.savefig("results.jpg")


def run_a_sample(netG, unetG, dataset, hps, training_typo, device):
    if not netG:
        netG = Unet_Generator(hps) if unetG else Generator(hps)
        state_dict = torch.load(f"{os.getcwd()}/scripts/trained_models/Generator{str(training_typo)}.pt",
                                map_location=device)
        netG.load_state_dict(state_dict)

    netG.to(device)

    noise = torch.randn(1, dataset.shape[1], dataset.shape[2], device=device) if unetG else torch.randn(1, 100, 1,
                                                                                                        device=device)
    output = netG(noise.to(device)).to(device)
    return output


def get_name(*argv):
    s = ''
    reduction = ''
    for arg in argv:
        try:
            if arg is not None:
                s += '-' + arg.__class__.__name__
            if arg.reduction:
                reduction = arg.reduction
        except:
            pass
    if reduction != '':
        s += '-' + reduction
    if '-DataParallel' in s:
        s = s.replace('-DataParallel', '-Generator')
    return s

def print_time(ss):
    """
    Print in hh:mm:ss format given duration.
    :param ss: given seconds.
    :return: print in display.
    """
    ss = int(ss)
    hh = 0
    mm = 0
    if ss>(60*60):
        hh = ss//(60*60)
        ss -= (ss//(60*60))*(60*60)
    if ss>60:
        mm = ss//60
        ss -= (ss//60)*60
    hh = str(int(hh)).zfill(2)
    mm = str(int(mm)).zfill(2)
    ss = str(int(ss)).zfill(2)
    return f"{bcolors.OKGREEN}{hh}:{mm}:{ss}{bcolors.ENDC}"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class WSConvTranspose1d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then
    it will multiply the scale for every forward pass
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
        super(WSConvTranspose1d, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)

        # new bias to use after wscale
        #self.bias = self.conv.bias
        #self.conv.bias = None

        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:])  # Leave out # of op filters
        self.wtScale = gain / np.sqrt(fanIn)

        # init
        nn.init.normal_(self.conv.weight)
        #nn.init.constant_(self.bias, val=0)

    def forward(self, x):
        return self.conv(x) * self.wtScale #+ self.bias.view(1, self.bias.shape[0], 1)


