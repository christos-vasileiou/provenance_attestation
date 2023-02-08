import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from gan_models import *
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import plotly.offline as pyo
import torch
from tqdm import tqdm
import pandas as pd
import logging

def plot_pca(trainSet, netG, maskset, iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = []
    for s in tqdm(range(len(trainSet))):
        # create samples from Up-CNN
        noise = torch.randn(1, 100, 1)
        samples.append(generate_sample(netG, noise, device))
    samples = np.array(samples)
    t = trainSet.copy().reshape(trainSet.shape[0], -1)
    samples = pd.DataFrame(samples.reshape(samples.shape[0], -1))
    logging.info(f"Are the samples unique? {is_unique(samples.to_numpy())}")
    dataset = trainSet.copy()
    dataset = pd.DataFrame(dataset.reshape(dataset.shape[0], -1))
    generated_pca = PCA(n_components=3).fit_transform(samples)
    real_pca = PCA(n_components=3).fit_transform(dataset)
    concatenated = PCA(n_components=3).fit_transform(pd.DataFrame(np.concatenate([t, samples])))
    trace1 = plot_3d_trace(real_pca, name='real')
    trace2 = plot_3d_trace(generated_pca, name='Up  BCE-PERC-HSphere maskset 0')
    trace3 = plot_3d_trace(concatenated[:t.shape[0]], name='Concatenated Real')
    trace4 = plot_3d_trace(concatenated[t.shape[0]:, ], name='Concatenated Gen')
    data = [trace1, trace2, trace3, trace4]

    fig = go.Figure(data=data)
    fig.update_layout(title='real vs synthetic | PCA', autosize=False, width=880, height=400, showlegend=True)
    fig.write_html(f"./QE/{maskset}maskset/{iter}/maskset{maskset}-{iter}.html")
    logging.info(f"3D PCA plot saved in: ./QE/{maskset}maskset/{iter}/maskset{maskset}.html")


def generate_sample(G, noise, device):
    G.to(device)
    noise = noise.to(device)
    output = G(noise).to(device)
    return output.cpu().detach().numpy()

def is_unique(s):
    #a = s.to_numpy() # s.values (pandas<0.24)
    if (s[0] == s).all():
        return "No"
    else:
        return "Yes"

def plot_3d_trace(x, color=None, name=''):
    component1, component2, component3 = x[:,0], x[:, 1], x[:, 2]
    autocolorscale = True if color else False
    trace = go.Scatter3d(x=component1,
                         y=component2,
                         z=component3,
                         mode='markers',
                         marker=dict(autocolorscale=autocolorscale,
                                     color=color),
                         name=name
                         )

    return trace

def convert_size(size):
    if size < 1024:
        return size
    elif (size >= 1024) and (size < (1024 * 1024)):
        return "%.2f KB" % (size / 1024)
    elif (size >= (1024 * 1024)) and (size < (1024 * 1024 * 1024)):
        return "%.2f MB" % (size / (1024 * 1024))
    else:
        return "%.2f GB" % (size / (1024 * 1024 * 1024))

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
        state_dict = torch.load('../../../scripts/trained_models/Generator' + str(training_typo) + '.pt',
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
    s += '-' + reduction
    return s

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


_parula_data = [[0.2081, 0.1663, 0.5292],
                [0.2116238095, 0.1897809524, 0.5776761905],
                [0.212252381, 0.2137714286, 0.6269714286],
                [0.2081, 0.2386, 0.6770857143],
                [0.1959047619, 0.2644571429, 0.7279],
                [0.1707285714, 0.2919380952, 0.779247619],
                [0.1252714286, 0.3242428571, 0.8302714286],
                [0.0591333333, 0.3598333333, 0.8683333333],
                [0.0116952381, 0.3875095238, 0.8819571429],
                [0.0059571429, 0.4086142857, 0.8828428571],
                [0.0165142857, 0.4266, 0.8786333333],
                [0.032852381, 0.4430428571, 0.8719571429],
                [0.0498142857, 0.4585714286, 0.8640571429],
                [0.0629333333, 0.4736904762, 0.8554380952],
                [0.0722666667, 0.4886666667, 0.8467],
                [0.0779428571, 0.5039857143, 0.8383714286],
                [0.079347619, 0.5200238095, 0.8311809524],
                [0.0749428571, 0.5375428571, 0.8262714286],
                [0.0640571429, 0.5569857143, 0.8239571429],
                [0.0487714286, 0.5772238095, 0.8228285714],
                [0.0343428571, 0.5965809524, 0.819852381],
                [0.0265, 0.6137, 0.8135],
                [0.0238904762, 0.6286619048, 0.8037619048],
                [0.0230904762, 0.6417857143, 0.7912666667],
                [0.0227714286, 0.6534857143, 0.7767571429],
                [0.0266619048, 0.6641952381, 0.7607190476],
                [0.0383714286, 0.6742714286, 0.743552381],
                [0.0589714286, 0.6837571429, 0.7253857143],
                [0.0843, 0.6928333333, 0.7061666667],
                [0.1132952381, 0.7015, 0.6858571429],
                [0.1452714286, 0.7097571429, 0.6646285714],
                [0.1801333333, 0.7176571429, 0.6424333333],
                [0.2178285714, 0.7250428571, 0.6192619048],
                [0.2586428571, 0.7317142857, 0.5954285714],
                [0.3021714286, 0.7376047619, 0.5711857143],
                [0.3481666667, 0.7424333333, 0.5472666667],
                [0.3952571429, 0.7459, 0.5244428571],
                [0.4420095238, 0.7480809524, 0.5033142857],
                [0.4871238095, 0.7490619048, 0.4839761905],
                [0.5300285714, 0.7491142857, 0.4661142857],
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143],
                [0.6473, 0.7456, 0.4188],
                [0.6834190476, 0.7434761905, 0.4044333333],
                [0.7184095238, 0.7411333333, 0.3904761905],
                [0.7524857143, 0.7384, 0.3768142857],
                [0.7858428571, 0.7355666667, 0.3632714286],
                [0.8185047619, 0.7327333333, 0.3497904762],
                [0.8506571429, 0.7299, 0.3360285714],
                [0.8824333333, 0.7274333333, 0.3217],
                [0.9139333333, 0.7257857143, 0.3062761905],
                [0.9449571429, 0.7261142857, 0.2886428571],
                [0.9738952381, 0.7313952381, 0.266647619],
                [0.9937714286, 0.7454571429, 0.240347619],
                [0.9990428571, 0.7653142857, 0.2164142857],
                [0.9955333333, 0.7860571429, 0.196652381],
                [0.988, 0.8066, 0.1793666667],
                [0.9788571429, 0.8271428571, 0.1633142857],
                [0.9697, 0.8481380952, 0.147452381],
                [0.9625857143, 0.8705142857, 0.1309],
                [0.9588714286, 0.8949, 0.1132428571],
                [0.9598238095, 0.9218333333, 0.0948380952],
                [0.9661, 0.9514428571, 0.0755333333],
                [0.9763, 0.9831, 0.0538]]