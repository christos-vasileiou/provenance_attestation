from absl import flags, app
from absl.flags import FLAGS
from wgan.gan_models import *
from torchinfo import summary
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from os.path import exists
from enum import Enum
import plotly.graph_objs as go
import random
from kneed import KneeLocator
from scipy.spatial import ConvexHull
from scipy.stats import wasserstein_distance
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
#import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import operator
import itertools
import math
import warnings
import pickle
import scipy
from sklearn.neighbors import KernelDensity
#from sklearn.model_selection import GridSearchCV
#from KDEpy import *
#from joblib.externals.loky.backend.context import get_context
import sys

flags.DEFINE_float('bandwidth',
                     1,
                     'KDE bandwidth')

def is_inside_radius(points, p, kernel, bandwidth, gen_no, mode='k-centers'):
    if mode == '1-centers':
        # 1 Center
        real_center            = points.mean(axis=0)
        real_to_real_closest_d = np.sqrt(np.sum((points - real_center) ** 2, axis=1))
        real_to_real_radius    = np.quantile(real_to_real_closest_d, 0.50)
        fig = plt.figure()
        plt.hist(real_to_real_closest_d, len(real_to_real_closest_d)//3, color='green')
        plt.savefig(f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen/real_to_real_closest_d.jpg")
        plt.close()
        if np.sqrt(np.sum((p-real_center) ** 2, axis=1)) < real_to_real_radius:
            return True
        else:
            return False
    elif mode == 'k-centers':
        # k Centers - based on KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4).fit(points)
        clustered_points = []
        for uniq_l in np.unique(kmeans.labels_):
            clustered_points.append(points[kmeans.labels_ == uniq_l])

        real_to_real_cluster_radiuses = []
        for cluster, center in zip(clustered_points, kmeans.cluster_centers_):
            real_to_real_closest_d = np.sqrt(np.sum((cluster - center) ** 2, axis=1))
            real_to_real_radius = np.quantile(real_to_real_closest_d, 0.8)
            real_to_real_cluster_radiuses.append(real_to_real_radius)

        for radius, center in zip(real_to_real_cluster_radiuses, kmeans.cluster_centers_):
            dist = np.sqrt(np.sum((p - center) ** 2, axis=1))
            if dist < radius:
                return True
        return False
    else:
        raise ValueError ('Set  `mode`  variable to `1-center` or `k-center`.')


def generate_synth_samples_w_kde(process_indx, data_points, t_m, master_list, kernel=None, bandwidth=None, gen_no=None, hps=None):
    trainset = data_points.iloc[:, 3:-2]
    trainset = trainset[master_list].values
    logging.info(f"kernel: {kernel}, bandwidth: {bandwidth}, trainset: {trainset.shape}")
    if kernel == None:
        raise ValueError(f"Error value in bandwidth: kernel: {kernel}")
    if bandwidth == None:
        bandwidth = FLAGS.bandwidth

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, metric="euclidean")
    kde.fit(trainset)

    pca = PCA(n_components=2)
    pca.fit(trainset)
    principalComponents = pca.transform(trainset)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1',
                                                                  'principal component 2'])
    principal_polygon = principalDf.values

    synth_samples = []
    synth_samples_2d = []
    while True:
        synth_sample = kde.sample(n_samples=1)
        if len(synth_samples) == 2*trainset.shape[0]:
            break
        synth_sample_2d_projected = pca.transform(synth_sample)
        #if is_inside_polygon(points=principal_polygon, p=synth_sample_2d_projected[0]):
        #    continue

        if is_inside_radius(points=trainset, p=synth_sample, kernel=kernel, bandwidth=bandwidth, gen_no=gen_no):
            continue

        synth_samples.append(synth_sample[0])
        synth_samples_2d.append(synth_sample_2d_projected[0])
    synth_samples = np.array(synth_samples)
    synth_samples_2d = np.array(synth_samples_2d)
    pd.DataFrame(synth_samples, columns=master_list).to_csv(f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen/synthetic_dataset_{process_indx}pid.csv")

    real_center = np.expand_dims(trainset.mean(axis=0), axis=0)
    real_to_real_closest_d = np.sqrt(np.sum((trainset - real_center) ** 2, axis=1))
    synth_to_real_closest_d = np.sqrt(np.sum((synth_samples - real_center) ** 2, axis=1))
    
    fig = plt.figure()
    plt.hist(real_to_real_closest_d, len(real_to_real_closest_d), color='blue')
    plt.hist(synth_to_real_closest_d, len(synth_to_real_closest_d), color='blue')
    plt.savefig(f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen/euclidean-distribution.jpg")
    plt.close()

    split = trainset.shape[0] / (trainset.shape[0] + synth_samples.shape[0])
    plot_3d_pca(pd.DataFrame(np.concatenate((trainset, synth_samples))), name=['orig', 'synth'], split=split,
                filename=f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen/prncComp_3d_polyBnd_wSynthSamples_{process_indx}pid.html")

    plot_2d_pca(pd.DataFrame(np.concatenate((principal_polygon, synth_samples_2d))), name=['orig', 'synth'], split=split,
                filename=f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen/prncComp_2d_polyBnd_wSynthSamples_{process_indx}pid.html")
    sys.exit()


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__ = state


def generate_synth_samples_w_timegan(process_indx, data_points, t_m, master_list, kernel=None, bandwidth=None, gen_no=None, hps=None):
    trainset = data_points.iloc[:, 3:-2]
    trainset = trainset[master_list]
    logging.info(f"TimeGAN: trainset: {trainset.shape}")
    if kernel == None:
        raise ValueError(f"Error value in bandwidth: kernel: {kernel}")
    if bandwidth == None:
        bandwidth = FLAGS.bandwidth

    ###########
    # Timegan training
    ###########

    from timegan.data.data_preprocess import data_preprocess
    from sklearn.model_selection import train_test_split
    from timegan.models.timegan import TimeGAN
    from timegan.models.utils import timegan_trainer, timegan_generator
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hps.update(
        {  # TimeGAN's parameters
            "padding_value": -1, "device": device, "exp": "test", "feat_pred_no": 1, "hidden_dim": 27,
            "max_seq_len": None, "batch_size": 32, "emb_epochs": 10000, "sup_epochs": 10000, "gan_epochs": 10000,
            "num_layers": 3, "dis_thresh": 0.15,
            "optimizer": "adam", "learning_rate": 1e-3})

    train_data, train_T, _, hps.max_seq_len, hps.padding_value = data_preprocess(
        file_name=None, max_seq_len=hps.max_seq_len, trainset=trainset
    )
    hps.update({"feature_dim": train_data.shape[-1], "Z_dim": train_data.shape[-1]})
    model = TimeGAN(hps)
    timegan_trainer(model, train_data, train_T, hps)

    ###########
    # End of training
    ###########

    pca = PCA(n_components=2)
    #print(trainset.shape)
    pca.fit(trainset)
    principalComponents = pca.transform(trainset)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1',
                                                                  'principal component 2'])
    principal_polygon = principalDf.values

    synth_samples = []
    synth_samples_2d = []
    # Check if you've sampled too many.
    while len(synth_samples) != 2 * trainset.shape[0]:
        # generate one sample called `synth_sample`
        #####################
        samples = timegan_generator(model, train_T, hps)
        samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))

        for i in range(len(samples)):
            # Check if you've sampled too many.
            if len(synth_samples) == 2 * trainset.shape[0]:
                break
            synth_sample = samples[i:i+1]
            synth_sample_2d_projected = pca.transform(synth_sample)

            # Check if the `synth_sample` falls inside the dense part of the population.
            if is_inside_radius(points=trainset, p=synth_sample, kernel=kernel, bandwidth=bandwidth, gen_no=gen_no):
                continue

            synth_samples.append(synth_sample[0])
            synth_samples_2d.append(synth_sample_2d_projected[0])

    synth_samples = np.array(synth_samples)
    synth_samples_2d = np.array(synth_samples_2d)
    # Before save the synthetic samples, reorder them to match the initial measurements.
    # synth_samples = postprocess()
    with open('../ordered_columns_list', 'r') as fr:
        ordered_columns = [line.split('\n')[0] for line in fr.readlines()]
    synth_samples = pd.DataFrame(synth_samples, columns=ordered_columns)
    data = synth_samples.reindex(master_list, axis=1)
    synth_samples.to_csv(f"{kernel}-k-clusters/{gen_no}gen/synthetic_dataset_{process_indx}pid.csv")


    """
    real_center = np.expand_dims(trainset.mean(axis=0), axis=0)
    print('\n', '\n', '\n', synth_samples.shape, real_center.shape)
    real_to_real_closest_d = np.sqrt(np.sum((trainset - real_center) ** 2, axis=1))
    synth_to_real_closest_d = np.sqrt(np.sum((synth_samples - real_center) ** 2, axis=1))

    fig = plt.figure()
    plt.hist(real_to_real_closest_d, len(real_to_real_closest_d), color='blue')
    plt.hist(synth_to_real_closest_d, len(synth_to_real_closest_d), color='blue')
    plt.savefig(f"{kernel}-k-clusters/{gen_no}gen/euclidean-distribution.jpg")
    plt.close()

    split = trainset.shape[0] / (trainset.shape[0] + synth_samples.shape[0])
    plot_3d_pca(pd.DataFrame(np.concatenate((trainset, synth_samples))), name=['orig', 'synth'], split=split,
                filename=f"{kernel}-k-clusters/{gen_no}gen/prncComp_3d_polyBnd_wSynthSamples_{process_indx}pid.html")

    plot_2d_pca(pd.DataFrame(np.concatenate((principal_polygon, synth_samples_2d))), name=['orig', 'synth'], split=split,
                filename=f"{kernel}-k-clusters/{gen_no}gen/prncComp_2d_polyBnd_wSynthSamples_{process_indx}pid.html")
    sys.exit()
    """
    
def generate_synth_samples_w_gans_old(process_indx, data_points, t_m, master_list, hps):
    trainset = data_points.iloc[:, 3:-2]
    trainset = trainset[master_list]
    logging.info("   HERE")
    # trainset dimensions (num of samples, features)
    samples = trainset.shape[0]
    features = trainset.shape[1]
    name = f"NSGA-WGAN-GP-{t_m}t_m" if hps['type'] == 1 else f"NSGA-WGAN-GP-{t_m}t_m"
    #wandb.init(project=f"NSGA-WGAN-{process_indx}pid", name=name, entity="chrivasileiou", config=hps)
    hps['osize'] = features
    hps['process_index'] = process_indx
    trainset = trainset.values

    logging.info(f"trainset: {trainset.shape}")
    # build the architectures
    generator = GeneratorInline(hps)
    discriminator = DiscriminatorInline(hps, wgan=True)

    # Prints out the summary of the architectures
    summary(generator)
    summary(discriminator)
    logging.info('############ start training ############')
    startTime = time.time()

    _, generator, discriminator, _ = train_inline(generator,
                                                    discriminator,
                                                    trainset,
                                                    t_m,
                                                    n_epochs=hps['n_epochs'],
                                                    batch_size=hps['batch_size'],
                                                    lr=hps['lr'],
                                                    weight_decay=hps['weight_decay'],
                                                    beta1=hps['beta1'],
                                                    workers=hps['workers'],
                                                    iter=hps['process_index'],
                                                    critic_iter=hps['critic_iter'],
                                                    first_time=False,
                                                    typo=hps['type'],
                                                    load_model=hps['load_model'],
                                                    max_gen=hps['max_gen'])

    endTime = time.time()
    logging.info('############ training has stopped ############')
    duration = endTime - startTime
    logging.info("Total Training's duration")
    logging.info(print_time(duration))
    #wandb.finish()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #generator = generator.to(device)
    synthetic_samples = []
    synthetic_set_2d = []
    # Generate points around the boundary
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(trainset)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1',
                                                                  'principal component 2'])
    principal_polygon = principalDf.values

    while True:
        noise = torch.randn(1, 50, 1, device=device, dtype=torch.float32)
        generated_sample = generator(noise.to(device)).transpose(2, 1).cpu().detach().numpy()
        if len(synthetic_samples) == 2*trainset.shape[0]:
            break

        generated_sample_2d_projected = pca.transform(generated_sample[0])
        #if is_inside_polygon(points=principal_polygon, p=generated_sample_2d_projected[0]):
        #    if print_INSIDE:
        #        print("INSIDE")
        #        print_INSIDE = False
        #    continue
        #else:
        #   print("OUTSIDE POLYGON")

        #print((euclidean_distance(generated_sample, trainset)).min(), trainset.std().sum())
        #if (generated_sample - trainset).min() <= trainset.std().sum():
        #logging.info(f"New Synthetic {len(synthetic_samples)}")
        synthetic_samples.append(generated_sample[0])
        synthetic_samples.append(generated_sample[0][0])

    synthetic_samples = np.array(synthetic_samples)
    print(f"Trainset: {trainset.shape}, Synthetic: {synthetic_samples.shape}, Concatenate: {np.concatenate((trainset, synthetic_samples)).shape}")
    split = trainset.shape[0]/(trainset.shape[0]+synthetic_samples.shape[0])
    plot_3d_pca(pd.DataFrame(np.concatenate((trainset, synthetic_samples))), name=['orig','synth'], split=split, filename=f"principalComp_polygonBound_wSynthSamples_{process_indx}pid.html")
    synthetic_set_2d = np.array(synthetic_set_2d)
    pd.DataFrame(synthetic_samples, columns=master_list).to_csv(f"synthetic_dataset_{process_indx}pid.csv")

def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def train_inline(netG, netD, trainset, t_m, n_epochs=100, batch_size=16, lr=0.001,
          weight_decay=0.14, beta1=0.5, workers=1, iter=None, LAMBDA_TERM=100, critic_iter=10, first_time=True, typo=None, load_model=None, max_gen=None):
    """Train the neural network.
    """
    time_zero = time.time()
    # seed and initialization
    torch.cuda.manual_seed_all(123456)
    torch.manual_seed(123456)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = netG.to(device)
    netD = netD.to(device)
    # Handle multi-gpu if desired
    #if (device.type == 'cuda') and (ngpu > 1):
    #    netG = nn.DataParallel(netG, list(range(ngpu)))
    #    netD = nn.DataParallel(netD, list(range(ngpu)))

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
    weight_clipping_limit = 1.
    gp = ' + LAMBDA*gp' if typo == 2 else ''

    # set up function loss, optimizer, and learning rate scheduler.
    training_typo = get_name(netG)
    training_typo += '-Inline'
    training_typo += '-' + str(iter) if iter else ''
    training_typo += f"-critic{str(critic_iter)}-epochs{n_epochs}"
    logging.info(f"Training type: {training_typo}")

    # Setup Adam optimizers for both G and D
    if typo == 1:
        optimizerG = optim.RMSprop(netG.parameters(), lr=lr, weight_decay=weight_decay)
        optimizerD = optim.RMSprop(netD.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9), weight_decay=weight_decay)
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9), weight_decay=weight_decay)

    # stores results over all epochs
    results = pd.DataFrame(index=list(range(1, n_epochs + 1)),
                           columns=["Train Loss", "Validation Loss", "Test Loss",
                                    "Min Validation Loss"])
    # load a model to train, if specified.
    if load_model and first_time:
        #filepath = f"../scripts/trained_models/InstanceNorm/{t_m}msks-Generator-Generator-BCELoss-sum-Etest-critic20-epochs4000-final.pt"
        filepath = f"../scripts/trained_models/InstanceNorm/{load_model}"
        netG.load_state_dict(torch.load(filepath))
        #netD.load_state_dict((torch.load(f"../scripts/trained_models/{t_m}msks-Discriminator{str(training_typo)}.pt")))
        #logging.info(f"{bcolors.OKGREEN}Trained model G loaded from the path: {filepath}{bcolors.ENDC}, for the t_m: {t_m}")
        logging.info(
            f"Trained model G loaded from the path: {filepath}, for the t_m: {t_m}")
    else:
        logging.info(f"Start Training from the scratch: t_m -> {t_m}, description -> {str(training_typo)} ")
        #logging.info(f"{bcolors.OKGREEN}Start Training from the scratch: t_m -> {t_m}, description -> {str(training_typo)} {bcolors.ENDC} ")

    #wandb.watch(netG, log="all", log_freq=5)
    #wandb.watch(netD, log="all", log_freq=5)
    #wandb.watch(netD, log="all", log_freq=5)
    wd = np.inf if typo == 2 else 0
    edmin = np.inf
    # Create the folder to store files.
    storage_path = f"../scripts/trained_models/NSGA-WGAN-CP/{iter}" if typo == 1 else f"../scripts/trained_models/NSGA-WGAN-GP/{iter}"
    if not exists(storage_path) and iter is not None:
        os.mkdir(storage_path)
    final_G_state_dict = f"{storage_path}/{t_m}msks-Generator{str(training_typo)}-final.pt"
    final_D_state_dict = f"{storage_path}/{t_m}msks-Discriminator{str(training_typo)}-final.pt"
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, multiprocessing_context=get_context('loky'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    for epoch in range(n_epochs + 1):
        iterations = len(trainset) // batch_size
        startTime = time.time()
        for i, data in enumerate(trainloader):
            ###################################################
            # (1) Update D network: maximize [D(x) - D(G(z))] #
            ###################################################
            for p in netG.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netD.parameters():
                p.requires_grad = True

            ## Train with all-real batch
            netD = netD.to(device)
            netG = netG.to(device)
            for d_iter in range(critic_iter):
                netD.zero_grad()
                # Format batch
                real = torch.unsqueeze(data, dim=1).to(torch.float32).to(device)
                b_size = real.size(0)
                if typo == 1:
                    for p in netD.parameters():
                        p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)
                if b_size == 1:
                    continue
                # Forward pass real batch through D
                real_outputs = netD(real)
                output = real_outputs[-1].view(-1)
                # print('OUTPUT D:' ,output.shape)
                # Calculate loss on all-real batch
                d_loss_real = output.mean()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, 50, 1, device=device)
                #print('NOISE: ', noise.shape)
                fake = netG(noise)

                # Classify all fake batch with D
                fake_outputs = netD(fake.transpose(2,1).detach())
                output = fake_outputs[-1].view(-1)
                # Calculate D's loss on the all-fake batch
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                d_loss_fake = output.mean()

                # Train with gradient penalty
                # calculate the Wasserstein loss
                d_loss = d_loss_fake - d_loss_real
                if typo == 2:
                    gradient_penalty = calculate_gradient_penalty(real, fake.transpose(2,1), netD)
                    d_loss += gradient_penalty * LAMBDA_TERM
                Wasserstein_loss = d_loss_fake - d_loss_real
                d_loss.backward(retain_graph=True)
                #d_loss_fake = d_loss_fake.mean()
                #d_loss_real = d_loss_real.mean()
                # Update D
                optimizerD.step()
                #print(f'  Discriminator iteration: {d_iter}/{critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            ############################################
            # (2) Update G network: minimize [-D(G(z)] #
            ############################################
            for p in netG.parameters():
                p.requires_grad = True
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation

            netG.zero_grad()

            if b_size == 1:
                continue
            # Since we just updated D, perform another forward pass of all-fake batch through D
            noise = torch.randn(data.shape[0], 50, 1, device=device)
            fake = netG(noise.to(device)).to(device)
            fake_outputs = netD(fake.transpose(2,1).to(device))
            output = fake_outputs[-1].view(-1)
            # Calculate G's loss based on this output
            # Calculate gradients for G
            g_loss = -output.mean()
            g_loss.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if (i % 50 == 0 or i == iterations) and epoch % 10 == 0:
                endTime = time.time()
                epochTimes.append(endTime - startTime)
                # Print Stats
                #tepoch.set_postfix_str(f"Mask-Set: {t_m}, {epochTimes[-1]:.2f}sec: -> [{epoch}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {Wasserstein_loss.item():4.4f}, Loss_G: {g_cost.item():4.4f}",
                #    refresh=True)
                #logging.info(f"{bcolors.BOLD}{epochTimes[-1]:.2f} sec{bcolors.ENDC}: Mask-Set: [{bcolors.BOLD}{t_m}{bcolors.ENDC}] -> [{bcolors.OKCYAN}{epoch}{bcolors.ENDC}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {bcolors.OKCYAN}{Wasserstein_loss.item():5.4f}{bcolors.ENDC}, Loss_G: {bcolors.OKCYAN}{g_loss.item():5.4f}{bcolors.ENDC}")
                logging.info(
                    f"{epochTimes[-1]:.2f} sec: Mask-Set: [{t_m}] -> Generation: [{iter}/{max_gen}] -> [{epoch}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {Wasserstein_loss.item():5.4f}, Loss_G: {g_loss.item():5.4f}")

                # Start count again
                startTime = time.time()

            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(Wasserstein_loss.item())
            D_real.append(d_loss_real.item())
            D_fake.append(d_loss_fake.item())
            log = {f'Train D Loss Total (fake - real{gp})': d_loss.item(),
                       'Train D Loss Real': d_loss_real.item(),
                       'Train D Loss Fake': d_loss_fake.item(),
                       'Train Wasserstein Distance (fake-real)': Wasserstein_loss.item()}
            if typo == 2:
                log.update({'Train Gradient Penalty': gradient_penalty.item()})
            #wandb.log(log)

        #########################################################
        # GAN's Evaluation:                                     #
        # 1. Wasserstein Distance,                              #
        # 2. Euclidean Distance between Centers of distribution #
        # 3. Standard Deviation of Real                         #
        # 4. Standard Deviation of Synthetic                    #
        #########################################################
        eval_data = []
        ref_data = []
        samples = []
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                noise = torch.randn(data.shape[0], 50, 1, device=device)
                batch = netG(torch.tensor(noise).to(device))
                for i_batch in batch:
                    samples.append(torch.squeeze(i_batch).cpu())
                g = netD(batch.transpose(2,1))[-1]
                g = g.to(torch.device("cpu"))
                for i_batch in g:
                    eval_data.append(torch.squeeze(i_batch))
                d = netD(torch.unsqueeze(data, dim=1).to(torch.float32).to(device))[-1]
                d = d.to(torch.device("cpu"))
                for i_batch in d:
                    ref_data.append(torch.squeeze(i_batch))
                del data

        eval_data = np.array([t.numpy() for t in eval_data])
        ref_data = np.array([t.numpy() for t in ref_data])
        if len(eval_data.shape) > 1:
            eval_data = eval_data.transpose(1, 0)
            ref_data = ref_data.transpose(1, 0)
            wdistance = 0
            for e, r in zip(eval_data, ref_data):
                wdistance += wasserstein_distance(e, r)
            wdistance = wdistance/len(eval_data)
        else:
            wdistance = wasserstein_distance(np.array(ref_data), np.array(eval_data))
        samples = np.array([t.numpy() for t in samples])
        euclidean_distance = np.max(np.sqrt((trainset.reshape(trainset.shape[0], -1).mean(0) - samples.reshape(samples.shape[0], -1).mean(0)) ** 2))
        #wandb.log({'Test Wasserstein Distance': wdistance,
        #           'Test Euclidean Distance': euclidean_distance,
        #           'Standard Deviation of Real': trainset.std(),
        #           'Standard Deviation of Synthetic': samples.std()})
        if epoch % 20 == 0 or epoch == n_epochs-1:
            logging.info(f"{len(eval_data)} eval_data, {len(ref_data)} ref_data")
            logging.info(f"Earth Mover's Distance: {wdistance:.2f}, Euclidean's Distance of the Centers of the Distributions (Real-Generated): {euclidean_distance:.2f}, Synth STD: {samples.std()}")
            #logging.info(f"Earth Mover's Distance: {bcolors.OKGREEN}{wdistance:.2f}{bcolors.ENDC}, Euclidean's Distance of the Centers of the Distributions (Real-Generated): {bcolors.OKGREEN}{euclidean_distance:.2f}{bcolors.ENDC}, Synth STD: {samples.std()}")

        if np.abs(euclidean_distance) < edmin:
            edmin = np.abs(euclidean_distance)

            """if typo == 1 and np.abs(euclidean_distance) < 2. and samples.std() > trainset.std()-.1 and samples.std() < trainset.std()+.6:
                torch.save(netG.cpu().state_dict(), f"{storage_path}/{t_m}msks-Generator{str(training_typo)}.pt")
                torch.save(netD.cpu().state_dict(), f"{storage_path}/{t_m}msks-Discriminator{str(training_typo)}.pt")
            elif typo == 2 and np.abs(euclidean_distance) < 2. and samples.std() > trainset.std()-.1 and samples.std() < trainset.std()+.6:
                torch.save(netG.cpu().state_dict(), f"{storage_path}/{t_m}msks-Generator{str(training_typo)}.pt")
                torch.save(netD.cpu().state_dict(), f"{storage_path}/{t_m}msks-Discriminator{str(training_typo)}.pt")
            """
        torch.save(netG.cpu().state_dict(), final_G_state_dict)
        torch.save(netD.cpu().state_dict(), final_D_state_dict)
        #if samples.std() < 1:
        #    break

    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracy During Training")
    plt.plot(D_real, label="real")
    plt.plot(D_fake, label="fakes")
    plt.plot(D_losses, label="Wasserstein-D Loss")
    plt.plot(G_losses, label="G Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    out_filepath = f"{storage_path}/{t_m}msks-GDLoss" + str(training_typo) + ".png"
    plt.savefig(out_filepath)
    #wandb.log({'GDLoss': wandb.Image(out_filepath)})
    # Save the model checkpoint
    torch.save(netG.cpu().state_dict(), final_G_state_dict)
    torch.save(netD.cpu().state_dict(), final_D_state_dict)
    logging.info(f"{bcolors.OKGREEN}Generator Model Saved in the location: {final_G_state_dict}{bcolors.ENDC}")
    logging.info(f"{bcolors.OKGREEN}Discriminator Model Saved in the location: {final_D_state_dict}{bcolors.ENDC}")

    return results, netG, netD, training_typo


#################
# SURAAG's Helper Functions for NSGA : Starts here
##################

def prepare_data_1(t_m, data):
    m_1_data = pd.DataFrame()
    m_2_data = pd.DataFrame()
    m_1_data = data[data['mask_data'] == t_m]
    m_2_data = data.drop(m_1_data.index)
    #for false_masks in f_m:
    #    m_2_data = m_2_data.append(data[data['mask_data'] == false_masks])
    return(m_1_data, m_2_data)

def prepare_data(t_m, f_m, dataset):
    m_1_data = pd.DataFrame()
    m_2_data = pd.DataFrame()
    #print(len(dataset[dataset.index.duplicated()]))
    m_1_data = dataset[dataset['mask_data'] == t_m]
    for false_masks in f_m:
        m_2_data = m_2_data.append(dataset[dataset['mask_data'] == false_masks])
    return(m_1_data, m_2_data)

#normalization
def normalise(train_data, test_data):
    #print("In Normalisation")
    train_data_x = train_data.iloc[:,:-1]#.values
    train_data_y = train_data.iloc[:,-1]#.values
    test_data_x = test_data.iloc[:,:-1]#.values
    test_data_y = test_data.iloc[:,-1]#.values
    train_mean = np.mean(train_data_x, axis=0)
    train_std = np.std(train_data_x, axis=0)
    train_norm = (train_data_x - train_mean) / train_std
    test_norm = (test_data_x - train_mean) / train_std
    #to check if any column doesnt have na values
    for cols in train_norm.columns:
        if train_norm[cols].isna().sum() >0:
            #print(cols)
            del train_norm[cols]
            del test_norm[cols]
    #print train_norm.shape

    return(train_norm,test_norm,train_data_y,test_data_y)

def train_test(dataset, t_m):  # ,d):
    # f_m = diff_function(t_m,d)
    f_m = []
    for values in range(0, 16):
        if values != t_m:
            f_m.append(values)
    mask_1_data, mask_2_data = prepare_data(t_m, f_m, dataset)
    mask_1_data = mask_1_data.iloc[:, :-1]
    mask_2_data = mask_2_data.iloc[:, :-1]  # has all the other mask data
    mask_1_data['Mask'] = 1
    mask_2_data['Mask'] = -1
    train_data = pd.DataFrame()
    train_data_2 = pd.DataFrame()
    lot_df = mask_1_data.groupby(by=["LOT_CHILD"], dropna=False)
    for lots, lot_data in lot_df:
        temp_data = lot_data.sample(frac=0.85)
        train_data = train_data.append(temp_data)
        train_data_2 = train_data_2.append(lot_data.drop(temp_data.index))

    # print(train_data.shape, train_data_2.shape)
    frames = [train_data_2, mask_2_data]
    test_data = pd.concat(frames)
    return (train_data, test_data)

def knee_elbow(list_distance):
    velocity = np.diff(list_distance)
    acceleration = np.diff(velocity)
    if np.mean(acceleration) > 0:
        return('convex')
    else:
        return('concave')

def trendline(L):
    if all(x<=y for x, y in zip(L, L[1:])):
        return 'increasing'
    else:
        return 'decreasing'

def polygonArea(vertices):
    #A function to apply the Shoelace algorithm
    numberOfVertices = len(vertices)
    sum1 = 0
    sum2 = 0

    for i in range(0,numberOfVertices-1):
        sum1 = sum1 + vertices[i][0] *  vertices[i+1][1]
        sum2 = sum2 + vertices[i][1] *  vertices[i+1][0]

    #Add xn.y1
    sum1 = sum1 + vertices[numberOfVertices-1][0]*vertices[0][1]
    #Add x1.yn
    sum2 = sum2 + vertices[0][0]*vertices[numberOfVertices-1][1]

    area = abs(sum1 - sum2) / 2
    return area


def error_calculator(process_indx, data, error_dataset, t_m, master_list, kernel, bandwidth, gen_no, hps):
    knob_list = []
    error_scribe = error_dataset.iloc[:, 0]
    error_dataset = error_dataset.iloc[:, 1:]
    # print(knob, type(knob))
    # print(error_dataset.shape, error_dataset.columns)
    neigh = NearestNeighbors(n_neighbors=int(error_dataset.shape[0] * 0.05))
    nbrs = neigh.fit(error_dataset.iloc[:, 0:-2])
    distances, indices = nbrs.kneighbors(error_dataset.iloc[:, 0:-2])
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    curve_type = knee_elbow(distances)
    direction_trend = trendline(distances)
    # print(curve_type, direction_trend)
    # print(type(distances))
    y = list(range(len(distances)))
    kn = KneeLocator(distances, y, S=1.0, curve=curve_type, direction=direction_trend, interp_method="interp1d")
    # print([kn.knee])
    eps_value = kn.knee  # y[kn.knee]
    if eps_value is None or eps_value <= 0:
        # print(eps_value)
        #print(error_dataset.columns)
        eps_value = 0.4

    # Code for DB scan
    dbscan = DBSCAN(eps=eps_value*2, min_samples=int(error_dataset.shape[0] * 0.10))  # int(error_dataset.shape[0]*0.10))
    dbscan.fit(error_dataset.iloc[:, 0:-2])
    labels = dbscan.labels_
    # print(list(set(labels)))
    error_dataset['Label'] = labels


    # error_dataset_subset_outlier = error_dataset[error_dataset['Label'] == -1]
    error = (error_dataset[error_dataset['Label'] == -1].shape[0] - (
    error_dataset[error_dataset['y_pred'] == -1].shape[0])) / error_dataset.shape[0]
    # error calculation ends here


    # print(error_dataset.columns, error_dataset.shape)
    unique_labels = np.unique(labels)
    index = np.argwhere(unique_labels == -1)
    unique_labels = np.delete(unique_labels, index)
    if (len(unique_labels)) > 0:
        bounding_area_points_DBSCAN = []
        try:
        #if True:
            for values in unique_labels:
                x_y = []
                error_dataset_subset = error_dataset[error_dataset['Label'] == values]
                # display(error_dataset_subset)
                # display(pd.DataFrame(error_dataset_subset))
                error_dataset_subset = error_dataset_subset.rename_axis('index1').reset_index()

                # Remove code from here
                points = error_dataset_subset.iloc[:, 1:-3].values


                #convex_start = time.time()
                hull = ConvexHull(points)
                hull_indices = np.unique(hull.simplices.flat)
                hull_points = points[hull_indices, :]

                """
                #convex_end = time.time()
                #print(f"Points: {points.shape}, Hull Points: {hull_points.shape}")
                #logging.info(print_time(convex_end-convex_start, False))

                #convex_start = time.time()
                #hull = cv2.convexHull(points)
                #hull_indices = np.unique(hull.simplices.flat)
                #hull_points = points[hull_indices, :]
                #convex_end = time.time()
                #print(f"Points: {points.shape}, Hull Points: {hull_points.shape}")
                #logging.info(print_time(convex_end-convex_start, True))

                """

                # print(hull_points)
                # print('_______________')
                point_indexes = hull.vertices
                # print(point_indexes)


                # hull_dataset = pd.DataFrame()
                # Create a DataFrame with the initial 252 features of the selected points of the Boundary.
                boundary_dataframe = pd.DataFrame()
                for index in point_indexes:
                    # print(hull[index,0])
                    scribe_value = error_scribe[error_dataset_subset.iloc[index, 0]]
                    boundary_data = data.loc[data['SCRIBE'] == scribe_value]
                    boundary_dataframe = boundary_dataframe.append(boundary_data, ignore_index=False)

                # Christos' code starts here
                if "timegan" in kernel:
                    generate_synth_samples_w_timegan(process_indx, boundary_dataframe, t_m, master_list, kernel, bandwidth, gen_no, hps)
                else:
                    generate_synth_samples_w_kde(process_indx, boundary_dataframe, t_m, master_list, kernel, bandwidth, gen_no, hps)
                # Christos' code ends here

                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(hull_points)
                principalDf = pd.DataFrame(data=principalComponents,
                                           columns=['principal component 1', 'principal component 2'])
                points = principalDf.values

                #x_y = list(zip(*[principalDf[col] for col in principalDf]))
                hull = ConvexHull(points)
                for value in range(0, len(points[hull.vertices, 0])):
                    co_ord = []
                    co_ord.append(points[hull.vertices[value], 0])
                    co_ord.append(points[hull.vertices[value], 1])
                    x_y.append(co_ord)
                bounding_area_points_DBSCAN.append(x_y)
            total_area = 0
            for boundary in bounding_area_points_DBSCAN:
                total_area = total_area + polygonArea(boundary)
        except scipy.spatial.qhull.QhullError:
            #logging.info(f"in except {process_indx}")
            total_area = 100
    else:
        total_area = 100
    bounding_area_points_SVM = []
    x_y = []
    error_dataset_subset = error_dataset[error_dataset['y_pred'] == 1]
    try:
        bounding_area_points_SVM = []
        x_y = []
        error_dataset_subset = error_dataset[error_dataset['y_pred'] == 1]
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(error_dataset_subset.iloc[:, 0:-3].values)
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        points = principalDf.values
        hull = ConvexHull(points)
        for value in range(0, len(points[hull.vertices, 0])):
            co_ord = []
            co_ord.append(points[hull.vertices[value], 0])
            co_ord.append(points[hull.vertices[value], 1])
            x_y.append(co_ord)
        bounding_area_points_SVM.append(x_y)
        svm_area = polygonArea(bounding_area_points_SVM[0])
    except:
        svm_area = 0
    if svm_area / total_area > 2:
        knob = svm_area / total_area
    elif svm_area / total_area > 6:
        knob = 3
    else:
        knob = 3

    # knob = total_svm_area / total_area
    #knob_list.append(knob)
    #knob_avg = Average(knob_list)
    return knob, error






def ml_algorithm(process_indx, data, train_norm, valid_norm, train_y, valid_y, nu_val, train_scribe, valid_scribe, t_m, master_list, kernel, bandwidth, gen_no, hps):
    warnings.filterwarnings('ignore')
    pred = svm.OneClassSVM(nu=nu_val, kernel="rbf", gamma="auto")
    pred.fit(train_norm)
    file_path = f"{kernel}-k-clusters/{gen_no}gen" if "timegan" in kernel else f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen"
    pickle.dump(pred, open(f"{file_path}/trained_{process_indx}pid.sav", 'wb'))
    with open(f"{file_path}/train_norm_{process_indx}pid.txt", 'w') as fw:
        for element in train_norm.columns:
            #print(element)
            fw.write(element+'\n')
    # pred_train_y = pred.predict(train_norm)
    # pred_valid_y = pred.predict(valid_norm)
    train_norm['y_pred'] = pred.predict(train_norm)
    valid_norm['y_pred'] = pred.predict(valid_norm)
    # print(train_norm, train_scribe)
    train_norm.insert(loc=0, column='SCRIBE', value=train_scribe)
    valid_norm.insert(loc=0, column='SCRIBE', value=valid_scribe)
    train_norm_parameters = train_norm.iloc[:, 1:-1]
    valid_norm_parameters = valid_norm.iloc[:, 1:-1]
    complete_train_data = pd.concat([train_norm, valid_norm])
    pred_complete_y = complete_train_data['y_pred']
    complete_y = pd.concat([train_y, valid_y])
    # pred_valid_y = valid_norm['y_pred']
    # valid_conf = confusion_matrix(valid_y, pred_valid_y)
    complete_conf = confusion_matrix(complete_y, pred_complete_y)
    pos, neg = 0, 0
    if len(complete_conf) > 1:
        pos = (complete_conf[1][1]) / complete_y.value_counts()[1]
        # error = abs((within_three_valid.shape[0] - valid_conf[1][1])/valid_y.value_counts()[1])
    else:
        pos = (complete_conf[0][0]) / complete_y.value_counts()[1]
        # error = abs((within_three_valid.shape[0] - valid_conf[0][0])/valid_y.value_counts()[1])
    # train_norm_mask = train_norm.copy(deep = True)
    # train_norm_mask['Mask'] = train_y
    complete_train_mask = complete_train_data.copy(deep=True)

    complete_train_mask['Mask'] = complete_y

    # print(train_norm_mask)
    std_list = []
    for values in complete_train_mask.std():
        std_list.append(values)
    # print(list(set(std_list)), len(list(set(std_list))))
    if list(set(std_list))[0]>0:  # Checking if all the columns have some variance
        knob, error = error_calculator(process_indx, data, complete_train_mask, t_m, master_list, kernel, bandwidth, gen_no, hps)
    else:
        error = 0.2 * pos
        knob = 1
    # print(pos, error)
    acc = pos - knob * error
    return (acc)

def generate_parents(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def generate_parents_random(inlist, n):
    list_1 = random.sample(inlist, n)
    list_1.insert(0, 'LOT_CHILD')
    list_1.append('mask_data')
    return(list_1)

def Average(lst):
    return sum(lst) / len(lst)

def function1(process_indx, x, data, t_m, nu_val, parentpath, master_list, kernel, bandwidth, gen_no, hps):
    #logging.info(f"{process_indx}, {parentpath}")
    x = x[process_indx]
    # weight = get_weight(x)
    if not ('LOT_CHILD' in x):
        x.insert(0, 'LOT_CHILD')
    else:
        if x[0] != 'LOT_CHILD':
            x.remove('LOT_CHILD')
            x.insert(0, 'LOT_CHILD')
    if not ('mask_data' in x):
        # print(x)
        # print("_________________________")
        x.append('mask_data')
        # print(x)
    else:
        if x[-1] != 'mask_data':
            x.remove('mask_data')
            x.insert(-1, 'mask_data')

    if not ('SCRIBE' in x):
        x.insert(1, 'SCRIBE')
    else:
        if x[1] != 'SCRIBE':
            x.remove('SCRIBE')
            x.insert(0, 'SCRIBE')
    data_process = data[x]
    train_data, test_data = train_test(data_process, t_m)  # ,d)
    inline_train = train_data
    valid_data = pd.DataFrame()
    lot_df = inline_train.groupby(by=["LOT_CHILD"], dropna=False)

    lot_list = inline_train['LOT_CHILD'].unique()
    fit_func_runs = []

    #for i in range(0, 4):
    random.shuffle(lot_list)
    train_lot_list = lot_list[:int(0.75 * len(lot_list))]
    valid_lot_list = lot_list[int(0.75 * len(lot_list)):]
    train_data = inline_train.loc[inline_train['LOT_CHILD'].isin(train_lot_list)]
    valid_data = inline_train.loc[inline_train['LOT_CHILD'].isin(valid_lot_list)]
    # print(train_data.shape, valid_data.shape)
    train_scribe = train_data['SCRIBE']
    valid_scribe = valid_data['SCRIBE']
    train_data = train_data.iloc[:, 2:]  # Removing LOT_CHILD and SCRIBE from dataset
    valid_data = valid_data.iloc[:, 2:]  # Removing LOT_CHILD and SCRIBE from dataset
    # inline_test = test_data.iloc[:,1:] #Removing LOT_CHILD from dataset
    train_norm, valid_norm, train_y, valid_y = normalise(train_data, valid_data)
    # train_norm, test_norm, train_y, test_y = normalise(inline_train, inline_test)


    fit_func = ml_algorithm(process_indx, data, train_norm, valid_norm, train_y, valid_y, nu_val, train_scribe, valid_scribe, t_m, master_list, kernel, bandwidth, gen_no, hps)

    #fit_func_runs.append(fit_func_val)
    # print(len(fit_func_runs),fit_func_runs)
    #fit_func = Average(fit_func_runs)
    # print(weight,fit_func, fit_func*weight)
    return process_indx, fit_func  # *weight

#Second function to optimize Need to change this function, for number of columns
def function2(x):
    value = len(x)*(-1)
    return value

#Function to find index of list. What does this do?
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values. What does this do?
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list#Function to sort by values

def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q]) or (values1[p] >= values1[q]) or (values1[p] > values1[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p]) or (values1[q] >= values1[p]) or (values1[q] > values1[p]):
                n[p] = n[p] + 1
            '''if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1'''
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        try:
            distance[k] = distance[k]+ (values1[sorted1[k+1]] - values1[sorted1[k-1]])/(max(values1)-min(values1))
        except ZeroDivisionError:
            distance[k] = distance[k] + 0
    for k in range(1,len(front)-1):
        try:
            distance[k] = distance[k]+ (values2[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
        except ZeroDivisionError:
            distance[k] = distance[k] + 0
    return distance

def crossover(a,b):
    #print(a,b)
    n = random.randint(1,len(a))
    child_1 = a[:n]+ b[n:]
    child_2 = b[:n]+ a[n:]
    return(child_1 , child_2)

def sort_master(in_dict, max_limit,check_list):
    sorted_d = dict( sorted(in_dict.items(), key=operator.itemgetter(1),reverse=True))
    out_dict = dict(itertools.islice(sorted_d.items(), max_limit))
    compare_list = list(out_dict.keys())
    #print(check_list, compare_list)
    diff_number = len(Diff(check_list, compare_list))
    #print(diff_number)
    return(out_dict, diff_number)

def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def myfunction():
    return 0.1

def mutation(fn1_val, sol):
    # non_zero_list = []
    zero_list = []
    child_list = []
    for g, function1_score in enumerate(fn1_val):
        if function1_score == 0:
            if 'LOT_CHILD' in sol[g]:
                sol[g].remove('LOT_CHILD')
            elif 'mask_data' in sol[g]:
                sol[g].remove('mask_data')
            zero_list.append(sol[g])
            # print(g, function2_values[g], solution[g])

    # print("Here is the values from mutation")
    # print(len(zero_list))
    # print(len(sol))
    if len(zero_list) > 1:
        for i in range(0, len(zero_list) - 1, 2):
            j = i + 1
            c1, c2 = crossover(zero_list[i], zero_list[j])
            child_list.append(c1)
            child_list.append(c2)
    # print("After addition")
    # print(len(child_list))
    sol = sol + child_list
    # print(len(sol))


"""
SURAAG's Helper Function for NSGA : Stops here ####
"""



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

def plot_2d_trace(x, color=None, name=''):
    component1, component2, = x[:,0], x[:, 1]
    autocolorscale = True if color else False
    trace = go.Scatter(x=component1,
                         y=component2,
                         mode='markers',
                         marker=dict(autocolorscale=autocolorscale,
                                     color=color),
                         name=name
                         )
    return trace

def plot_3d_pca(df: pd.DataFrame, name, split = 1, filename: str = "default"):
    """
    :param df:       DataFrame with all the data concatenated together.
    :param name:     Give the names sorted with the same way that have been concatenated together.
                     You need to pass that many names as the parts of the dataset.
    :param split:    Give the ratio of each portion of dataset.
                     You need to pass as many as ${names-1} for split.
                     If you pass a list of splits, then you must pass a list of names
    :param filename: filename for 2d PCA plot .html
    :return:
    """
    values = []
    if split == 1:
        values = df.values
    else:
        if isinstance(split, float):
            pivot = int(df.values.shape[0] * split)
            #print(pivot)
            values.append(df.values[:pivot])
            values.append(df.values[pivot:])
        elif isinstance(split, list):
            prev_pivot = 0
            for s in split:
                pivot = int(df.values.shape[0] * split)
                values.append(df.values[prev_pivot:pivot])
                prev_pivot = pivot

    pca = PCA(n_components=3).fit(values[0])

    data = []
    if isinstance(values, list):
        for val, val_name in zip(values, name):
            pca_values = pca.transform(val)
            data.append(plot_3d_trace(pca_values, name=val_name))
    else:
        pca_values = pca.transform(values)
        data.append(plot_3d_trace(pca_values, name=name))
    fig = go.Figure(data=data)
    fig.update_layout(title=f'3D Projection of Samples ({name})', autosize=False, width=680, height=600, showlegend=True)
    fig.write_html(filename)
    #fig.show()
    return fig

def plot_2d_pca(df: pd.DataFrame, name, split = 1, filename = "default"):
    """
    :param df:       DataFrame with all the data concatenated together.
    :param name:     Give the names sorted with the same way that have been concatenated together.
                     You need to pass that many names as the parts of the dataset.
    :param split:    Give the ratio of each portion of dataset.
                     You need to pass as many as ${names-1} for split.
                     If you pass a list of splits, then you must pass a list of names
    :param filename: filename for 2d PCA plot .html
    :return:
    """
    values = []
    if split == 1:
        values = df.values
    else:
        if isinstance(split, float):
            pivot = int(df.values.shape[0] * split)
            #print(pivot)
            values.append(df.values[:pivot])
            values.append(df.values[pivot:])
        elif isinstance(split, list):
            prev_pivot = 0
            for s in split:
                pivot = int(df.values.shape[0] * split)
                values.append(df.values[prev_pivot:pivot])
                prev_pivot = pivot

    pca = PCA(n_components=2).fit(values[0])

    data = []
    if isinstance(values, list):
        for val, val_name in zip(values, name):
            pca_values = pca.transform(val)
            data.append(plot_2d_trace(pca_values, name=val_name))
    else:
        pca_values = pca.transform(values)
        data.append(plot_2d_trace(pca_values, name=name))

    fig = go.Figure(data=data)
    fig.update_layout(title=f'2D Projection of Samples ({name})', autosize=False, width=680, height=600, showlegend=True)
    fig.write_html(filename)
    #fig.show()
    return fig

# Define Infinite (Using INT_MAX caused overflow problems)
INT_MAX = 10000
# Given three collinear points p, q, r, the function checks if point q lies on line segment 'pr'
def onSegment(p, q, r) -> bool:

	if ((q[0] <= max(p[0], r[0])) &
		(q[0] >= min(p[0], r[0])) &
		(q[1] <= max(p[1], r[1])) &
		(q[1] >= min(p[1], r[1]))):
		return True

	return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p, q, r) -> int:

	val = (((q[1] - p[1]) *
			(r[0] - q[0])) -
		((q[0] - p[0]) *
			(r[1] - q[1])))

	if val == 0:
		return 0
	if val > 0:
		return 1 # Collinear
	else:
		return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):

	# Find the four orientations needed for
	# general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if (o1 != o2) and (o3 != o4):
		return True

	# Special Cases
	# p1, q1 and p2 are collinear and
	# p2 lies on segment p1q1
	if (o1 == 0) and (onSegment(p1, p2, q1)):
		return True

	# p1, q1 and p2 are collinear and
	# q2 lies on segment p1q1
	if (o2 == 0) and (onSegment(p1, q2, q1)):
		return True

	# p2, q2 and p1 are collinear and
	# p1 lies on segment p2q2
	if (o3 == 0) and (onSegment(p2, p1, q2)):
		return True

	# p2, q2 and q1 are collinear and
	# q1 lies on segment p2q2
	if (o4 == 0) and (onSegment(p2, q1, q2)):
		return True

	return False

# Returns true if the point p lies
# inside the polygon[] with n vertices
def is_inside_polygon(points, p) -> bool:

	n = len(points)

	# There must be at least 3 vertices
	# in polygon
	if n < 3:
		return False

	# Create a point for line segment
	# from p to infinite
	extreme = (INT_MAX, p[1])

	# To count number of points in polygon
	# whose y-coordinate is equal to
	# y-coordinate of the point
	decrease = 0
	count = i = 0

	while True:
		next = (i + 1) % n

		if(points[i][1] == p[1]):
			decrease += 1

		# Check if the line segment from 'p' to
		# 'extreme' intersects with the line
		# segment from 'polygon[i]' to 'polygon[next]'
		if (doIntersect(points[i],
						points[next],
						p, extreme)):

			# If the point 'p' is collinear with line
			# segment 'i-next', then check if it lies
			# on segment. If it lies, return true, otherwise false
			if orientation(points[i], p,
						points[next]) == 0:
				return onSegment(points[i], p,
								points[next])

			count += 1

		i = next

		if (i == 0):
			break

	# Reduce the count by decrease amount
	# as these points would have been added twice
	count -= decrease

	# Return true if count is odd, false otherwise
	return (count % 2 == 1)


class T(Enum):
    clipping = 1
    gradient_penalty = 2

def pop_n_items(trainSet: np.ndarray, samples: np.ndarray, pop_items: int):
    average_point_from_gan = samples.mean(axis=0)
    distance = np.sqrt((trainSet - average_point_from_gan) ** 2)
    # take the indices sorted in ascending order, based on the distance.
    sort = np.sum(distance, axis=1).argsort()
    # .reshape((trainSet.shape[0], -1))
    trainSet = list(trainSet)
    j = 0
    poplist = sort[:pop_items]
    for i in sorted(poplist):
        # logging.info(i)
        trainSet.pop(i - j)
        j = i
    trainSet = np.array(trainSet)
    return trainSet

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

def print_time(sss, colored=False):
    """
    Print in hh:mm:ss format given duration.
    :param ss: given seconds.
    :return: print in display.
    """
    ss = int(sss)
    sss = sss - int(sss)
    hh = 0
    mm = 0
    if ss>(60*60):
        hh = ss//(60*60)
        ss -= (ss//(60*60))*(60*60)
    if ss>60:
        mm = ss//60
        ss -= (ss//60)*60
    sss = str(sss).split('.')[1]
    hh = str(int(hh)).zfill(2)
    mm = str(int(mm)).zfill(2)
    ss = str(int(ss)).zfill(2)
    if colored:
        return f"{bcolors.OKGREEN}{hh}:{mm}:{ss}.{sss}{bcolors.ENDC}"
    else:
        return f"{hh}:{mm}:{ss}.{sss}"

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


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

