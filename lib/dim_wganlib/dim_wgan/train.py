from absl import flags, app
from absl.flags import FLAGS
import torch.optim as optim
from matplotlib import pyplot
import pandas as pd
import numpy as np
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import time, statistics
import scipy.io as sio
from gan_models import *
import os
import logging
from tqdm import tqdm
import warnings

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

flags.DEFINE_boolean('perc',
                    False,
                    'set if perceptual loss will be applied.')

flags.DEFINE_boolean('p2p',
                    False,
                    'set if pixel-wise loss will be applied between fakes and ideals.')

flags.DEFINE_string('dataset',
                    '../../../scripts/data/gf_4masks_data_format_samples_features_sites.mat',
                    'Set the dataset with real labels.')

def convert_size(size):
    if size <1024:
        return size
    elif (size >= 1024) and (size < (1024 * 1024)):
        return "%.2f KB"%(size/1024)
    elif (size >= (1024*1024)) and (size < (1024*1024*1024)):
        return "%.2f MB"%(size/(1024*1024))
    else:
        return "%.2f GB"%(size/(1024*1024*1024))

def is_unique(u):
    #a = s.to_numpy() # s.values (pandas<0.24)
    return (u[0] == u).all()

def generate_sample(G, noise, device, dtype):
    G.to(device)
    noise = noise.to(device)
    #noise = torch.randn(1, dataset.shape[1], dataset.shape[2], device=device)
    #noise = torch.randn(1, 100, 1, device=device)

    output = G(noise).to(device)
    return output.cpu().detach().numpy()

def plotResults(results):
    fig, [ax1, ax2] = pyplot.subplots(2, 1, sharex=True, figsize=(10, 10))

    ax1.plot(results.index, results["Train Loss"],
             results.index, results["Test Loss"],
             results.index, results["Validation Loss"],
             results.index, results["Min Validation Loss"])
    ax1.legend(["Train", "Test", "Validation", "Min Validation"])
    ax1.set_title("Loss")

    ax2.plot(results.index, results["Validation Loss"] / results["Min Validation Loss"] - 1)
    ax2.set_title("Val Loss/Min Val Loss - 1")

    pyplot.savefig("results.jpg")


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
                s += '-'+arg.__class__.__name__
            if arg.reduction:
                reduction = arg.reduction
        except:
            pass
    s += '-'+reduction
    return s

def train(netG, netD, trainSet, validationSet=None, testSet=None, n_epochs=100, batch_size=16, lr=0.001, schedFactor=0.1, schedPatience = 10, weight_decay = 0.14, beta1 = 0.5, workers = 1, ngpu = 1, unetG=True, etestG=True, device = None, iter=None):
    """Train the neural network.
       
       Args:
           X (RadarDataSet): training data set in the PyTorch data format
           Y (RadarDataSet): test data set in the PyTorch data format
           n_epochs (int): number of epochs to train for
           learning_rate: starting learning rate (to be decreased over epochs by internal scheduler
    """
    os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "2"  #to limit thread usage by the numpy library

    # seed and initialization
    torch.cuda.manual_seed_all(123456)
    torch.manual_seed(123456)
    #torch.autograd.set_detect_anomaly(True)

    netG = netG.to(device)
    netD = netD.to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_fake = []
    D_real = []
    D_binary = []
    D_perc = []
    iters = 0
    epochTimes = []
    my_images = []

    # set up function loss, optimizer, and learning rate scheduler.
    criterion_p2p = None
    criterion_min_max = None
    criterion_perc = None
    criterion_min_max = nn.BCELoss(reduction='sum').to(device)
    criterion_perc = nn.L1Loss(reduction='sum').to(device) if FLAGS.perc else None
    criterion_p2p = nn.L1Loss(reduction='sum').to(device) if FLAGS.p2p else None
    training_typo = get_name(netG, criterion_min_max, criterion_perc, criterion_p2p)
    if iter:
        training_typo += '-'+str(iter)
    logging.info(f"Training type: {training_typo}")

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay = weight_decay)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay = weight_decay)
    
    #schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    #schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    
    # stores results over all epochs
    results = pd.DataFrame(index = list(range(1, n_epochs + 1)), 
                            columns = ["Train Loss", "Validation Loss", "Test Loss",
                                       "Min Validation Loss"])
    # load a model to train, if specified.
    #if FLAGS.load_model:
        #pass #self.load_state_dict(torch.load(FLAGS.load_model))
    
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=workers)
    # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU]
    # with profile(activities=activities, profile_memory=True) as prof:
    for epoch in range(n_epochs+1):
        iterations = len(trainSet)//batch_size
        startTime = time.time()
        with tqdm(trainloader, unit='batch') as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}:")
                ###############################################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
                #                  and  minimize L1Loss(fake_xi, real_xi)     #
                ###############################################################

                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                targets = data.to(device)
                b_size = targets.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
                #print('targets: ', targets.shape, label.shape)

                # Forward pass real batch through D
                real_outputs = netD(targets)
                output = real_outputs[-1].view(-1)
                #print('OUTPUT D:' ,output.shape)
                # Calculate loss on all-real batch
                errD_real = criterion_min_max(output, label) / 2.
                # Calculate gradients for D in backward pass
                if criterion_min_max.reduction == 'none':
                    for err in errD_real:
                        err.backward(retain_graph=True)
                else:
                    errD_real.backward(retain_graph=True)
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2], device=device) if unetG else torch.randn(b_size, 100, 1, device=device)
                noise = noise if etestG else torch.randn(b_size, 249, 1, 1, device=device)
                #print('NOISE: ', noise.shape)
                # Generate fake image batch with G
                fake = netG(noise.to(device)).to(device)
                label.fill_(fake_label)
                # Classify all fake batch with D
                fake_outputs = netD(fake.detach())
                output = fake_outputs[-1].view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion_min_max(output, label) / 2.
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                if criterion_min_max.reduction == 'none':
                    for err in errD_fake:
                        err.backward(retain_graph=True)
                else:
                    errD_fake.backward(retain_graph=True)
                D_G_z1 = output.mean().item()

                # Calculate D's perceptual loss which penalize the discrepancy between intermediate feature maps extracted by D.
                if criterion_perc:
                    fake_outputs = netD(fake)
                    errD_perc = 0
                    k=0
                    # TODO: FIX the lambda variable
                    lambda_var = [.001 for _ in range(3)]
                    for ro, fo, l in zip(real_outputs, fake_outputs, lambda_var):
                        k+=1
                        if k == 3:
                            break
                        #print(ro.size(), fo.size())
                        errD_perc_layer = criterion_perc(fo, ro)
                        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                        if criterion_perc.reduction == 'none':
                            errD_perc_layer = errD_perc_layer.sum(dim=2).sum(dim=1)
                            for err in errD_perc_layer:
                                err.backward(retain_graph=True)
                        else:
                            errD_perc_layer *= l
                            errD_perc_layer.backward(retain_graph=True)
                        errD_perc += errD_perc_layer.sum(dim=0)
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                if criterion_perc:
                    errD += errD_perc
                # Update D
                optimizerD.step()

                ###############################################
                # (2) Update G network: maximize log(D(G(z))) #
                ###############################################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2], device=device) if unetG else torch.randn(b_size, 100, 1, device=device)
                noise = noise if etestG else torch.randn(b_size, 249, 1, 1, device=device)
                fake = netG(noise.to(device)).to(device)
                fake_outputs = netD(fake.to(device))
                output = fake_outputs[-1].view(-1)
                # Calculate G's loss based on this output
                errG = criterion_min_max(output, label)
                # Calculate gradients for G
                if criterion_min_max.reduction == 'none':
                    for err in errG:
                        err.backward(retain_graph=True)
                else:
                    errG.backward(retain_graph=True)

                if criterion_p2p:
                    errG_p2p = criterion_p2p(fake, data.to(device)) / .01
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    if criterion_p2p.reduction == 'none':
                        for err in errG_p2p:
                            err.backward(retain_graph=True)
                    else:
                        errG_p2p.backward(retain_graph=True)
                    errG += errG_p2p
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()


                # Output training stats
                if i % 50 == 0 or i==iterations:
                    epochTimes.append(time.time() - startTime)
                    # Try one sample
                    fake = netG(noise.to(device)).to(device)
                    fake_outputs = netD(fake.to(device))
                    output = fake_outputs[-1].view(-1)
                    certainty = output.mean().item()

                    # Print Stats
                    if criterion_min_max.reduction == 'none':
                        tepoch.set_postfix_str(
                            f'{epochTimes[-1]:.2f}sec: -> [{epoch}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {errD.sum().item():4.4f}, Loss_G: {errG.sum().item():4.4f}, D(x): {D_x:4.4f}, D(G(z)): {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:4.4f}',
                            refresh=True)
                    else:
                        tepoch.set_postfix_str(
                            f'{epochTimes[-1]:.2f}sec: -> [{epoch}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {errD.item():4.4f}, Loss_G: {errG.item():4.4f}, D(x): {D_x:4.4f}, D(G(z)): {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:4.4f}',
                            refresh=True)

                    # If Discriminator has no idea then break
                    if certainty >= 0.45 and certainty <= 0.55 and epoch >= n_epochs:
                        break
                    # Start count again
                    startTime = time.time()

                # Save Losses for plotting later
                if criterion_min_max.reduction == 'none':
                    G_losses.append(errG.sum().item())
                    D_losses.append(errD.sum().item())
                    D_real.append(D_x.mean())
                    D_fake.append(D_G_z2.mean())
                    D_binary.append((errD_real + errD_fake).sum().item())
                    if criterion_perc:
                        D_perc.append(errD_perc.sum().item())
                else:
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())
                    D_real.append(D_x)
                    D_fake.append(D_G_z2)
                    D_binary.append((errD_real + errD_fake).item())
                    if criterion_perc:
                        D_perc.append(errD_perc.item())

    #prof.export_chrome_trace("./trace" + str(training_typo) + ".json")
    #if torch.cuda.is_available():
    #    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    #else:
    #    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    pyplot.figure(figsize=(10,5))
    pyplot.title("Generator and Discriminator Loss During Training")
    pyplot.plot(G_losses,label="G")
    pyplot.plot(D_losses,label="D")
    pyplot.plot(D_binary, label='D binary')
    pyplot.plot(D_perc, label='D perc')
    pyplot.xlabel("iterations")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig('../../../scripts/trained_models/GDLoss'+str(training_typo)+'.png')

    pyplot.figure(figsize=(10, 5))
    pyplot.title("Discriminator Accuracy During Training")
    pyplot.plot(D_real, label="real")
    pyplot.plot(D_fake, label="fakes")
    pyplot.xlabel("iterations")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig('../../../scripts/trained_models/Daccuracy' + str(training_typo) + '.png')
    # Save the model checkpoint
    torch.save(netG.state_dict(), '../../../scripts/trained_models/Generator'+str(training_typo)+'.pt')
    
    return results, netG, netD, training_typo

class RMSELoss(nn.Module):
    def __init__(self, eps = 1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat,y)+self.eps)
        return loss

def main(*args):
    FLAGS = flags.FLAGS
    warnings.filterwarnings('ignore')
    if isinstance(args[0], str):
        dataset = args[0]
    else:
        dataset = FLAGS.dataset
    torch.backends.cudnn.deterministic = True
    cuda = True

    #device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load Dataset
    etestG = False # Select whether etest or inline dataset will be used. default: True
    trainSet = sio.loadmat(dataset)['gf_data'] if etestG else np.array(pd.read_csv(
        '../../../scripts/data/inline_maskset_8181x249.csv', index_col=[0]).drop(columns=['maskset']))
    logging.info(f"{trainSet.shape}\n{trainSet}")
    try:
        depth = FLAGS.depth
        nchan = FLAGS.nchan
        batch_size = FLAGS.batch_size
        ndf = FLAGS.ndf
        n_epochs = FLAGS.n_epochs
        learning_rate = FLAGS.learning_rate
        schedFactor = FLAGS.schedFactor
        schedPatience = FLAGS.schedPatience
        weight_decay = FLAGS.weight_decay
    except:
        depth = 2
        nchan = 64
        batch_size = 16
        ndf = 64
        n_epochs = 5
        learning_rate = 0.001
        schedFactor = .1
        schedPatience = 3
        weight_decay = .14
        print("An Error was raised by trying to parse FLAGS.")

    # hps: hyperparameters
    hps = {
        'depth': depth,
        'nchan': nchan,
        # batch size during training
        'batch_size': batch_size,
        # Number of workers for dataloader
        'workers': 2,
        # Number of channels in the training images.
        'nc': trainSet.shape[1],
        # Size of feature maps in discriminator
        'ndf': ndf,
        # Beta1 hyperparam for Adam optimizers
        'beta1': 0.5,
        # Number of GPUs available. Use 0 for CPU mode.
        'ngpu': 1
    }

    unetG = False # Matters ONLY if 'etest' flag is true.
    resnetG = False
    netG = Unet_Generator(hps) if unetG else Generator(hps)
    netG = ResNetGenerator(hps) if resnetG else netG
    # if wants to generates inline dataset.
    netG = netG if etestG else GeneratorInline(hps)
    # Specify Discriminator
    netD = Discriminator(hps)
    
    logging.info('\n')
    print(netG)
    print(netD)
    iter=0
    print('############ start training ############')
    results, netG, netD, training_type = train(netG,
                                            netD,
                                            trainSet,
                                            n_epochs = n_epochs,
                                            batch_size = batch_size,
                                            lr = learning_rate,
                                            schedFactor = schedFactor,
                                            schedPatience = schedPatience,
                                            weight_decay = weight_decay,
                                            beta1 = hps['beta1'],
                                            workers = hps['workers'],
                                            ngpu = hps['ngpu'],
                                            unetG = unetG,
                                            etestG = etestG,
                                            device = device,
                                            iter = iter)
    print('############ end of training ############')
    #run_a_sample(None, unetG, trainSet, hps, training_type, device)

    n_gener_samples = 100
    pop_items = 200
    print("trainSet: ", trainSet.shape)
    initial_trainSet = trainSet.copy()
    #if FLAGS.perc:
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    while (len(list(trainSet)) >= 0):
        samples = []
        for _ in tqdm(range(n_gener_samples)):
            noise = torch.randn(1, 100, 1)
            samples.append(generate_sample(netG, noise, device, dtype))
        samples = np.array(samples)
        # make them 2d arrays
        samples = samples.reshape((len(samples), -1))
        trainSet = trainSet.reshape((len(trainSet), -1))
        print("Are the samples unique? ", is_unique(samples), " with mean: ", samples.mean(), " and std: ", samples.std())
        # calculate a point in a multi-dimensional space with the average values across the generated samples.
        average_point_from_gan = samples.mean(axis=0)
        # euclidean distance in a multi-dimensional space.
        distance = np.sqrt((trainSet - average_point_from_gan) ** 2)
        # take the indices sorted in ascending order, based on the distance.
        sort = np.sum(distance, axis=1).argsort()
        # .reshape((trainSet.shape[0], -1))
        trainSet = list(trainSet)
        j=0
        poplist = sort[:pop_items]
        for i in sorted(poplist):
            # print(i)
            trainSet.pop(i - j)
            j = i
        if len(trainSet) < pop_items:
            break
        iter += 1
        trainSet = np.array(trainSet).reshape((-1,393,13))
        print("trainSet: ", trainSet.shape)

        print('############ start training ############')
        results, netG, netD, training_type = train(netG,
                                                   netD,
                                                   trainSet,
                                                   n_epochs=n_epochs,
                                                   batch_size=batch_size,
                                                   lr=learning_rate,
                                                   schedFactor=schedFactor,
                                                   schedPatience=schedPatience,
                                                   weight_decay=weight_decay,
                                                   beta1=hps['beta1'],
                                                   workers=hps['workers'],
                                                   ngpu=hps['ngpu'],
                                                   unetG=unetG,
                                                   etestG=etestG,
                                                   device=device,
                                                   iter=iter)
        print('############ end of training ############')


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
