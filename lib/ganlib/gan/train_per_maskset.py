from absl import flags, app
from absl.flags import FLAGS
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import time, statistics
import scipy.io as sio
from gan_models import *
from utils import *
from metrics import compute_prd_from_embedding
import os
import logging
from tqdm import tqdm
import warnings
import wandb
from torchinfo import summary
from os.path import exists

flags.DEFINE_boolean('perc',
                     False,
                     'set if perceptual loss will be applied.')

flags.DEFINE_boolean('p2p',
                     False,
                     'set if pixel-wise loss will be applied between fakes and ideals.')

flags.DEFINE_string('dataset',
                    '../../../scripts/data/gf_4masks_data_format_samples_features_sites.mat',
                    'Set the dataset with real labels.')


def train(netG, netD, trainSet, maskset, validationSet=None, testSet=None, n_epochs=100, batch_size=16, lr=0.001,
          weight_decay=0.14, beta1=0.5, workers=1, ngpu=1, unetG=True, etestG=True,
          device=None, iter=None):
    """Train the neural network.

       Args:
           X (RadarDataSet): training data set in the PyTorch data format
           Y (RadarDataSet): test data set in the PyTorch data format
           n_epochs (int): number of epochs to train for
           learning_rate: starting learning rate (to be decreased over epochs by internal scheduler
    """

    # seed and initialization
    torch.cuda.manual_seed_all(123456)
    torch.manual_seed(123456)
    # torch.autograd.set_detect_anomaly(True)

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
        training_typo += '-' + str(iter)
    logging.info(f"Training type: {training_typo}")

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)

    # stores results over all epochs
    results = pd.DataFrame(index=list(range(1, n_epochs + 1)),
                           columns=["Train Loss", "Validation Loss", "Test Loss",
                                    "Min Validation Loss"])
    # load a model to train, if specified.
    # if FLAGS.load_model:
    # pass #self.load_state_dict(torch.load(FLAGS.load_model))

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=workers)
    wandb.watch(netG, log="all", log_freq=5)
    wandb.watch(netD, log="all", log_freq=5)
    for epoch in range(n_epochs + 1):
        iterations = len(trainSet) // batch_size
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
                real = data.to(device)
                b_size = real.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
                # print('real: ', real.shape, label.shape)

                # Forward pass real batch through D
                real_outputs = netD(real)
                output = real_outputs[-1].view(-1)
                # print('OUTPUT D:' ,output.shape)
                # Calculate loss on all-real batch
                errD_real = criterion_min_max(output, label) / 2.
                # Calculate gradients for D in backward pass
                errD_real.backward(retain_graph=True)
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2],
                                    device=device) if unetG else torch.randn(b_size, 100, 1, device=device)
                noise = noise if etestG else torch.randn(b_size, 249, 1, 1, device=device)
                # print('NOISE: ', noise.shape)
                # Generate fake image batch with G
                fake = netG(noise.to(device)).to(device)
                label.fill_(fake_label)
                # Classify all fake batch with D
                fake_outputs = netD(fake.detach())
                output = fake_outputs[-1].view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion_min_max(output, label) / 2.
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward(retain_graph=True)
                D_G_z1 = output.mean().item()

                # Calculate D's perceptual loss which penalize the discrepancy between intermediate feature maps extracted by D.
                if criterion_perc:
                    fake_outputs = netD(fake)
                    errD_perc = 0
                    k = 0
                    # TODO: FIX the lambda variable
                    lambda_var = [.001 for _ in range(3)]
                    for ro, fo, lv in zip(real_outputs, fake_outputs, lambda_var):
                        k += 1
                        if k == 3:
                            break
                        # logging.info(ro.size(), fo.size())
                        errD_perc_layer = criterion_perc(fo, ro) * lv
                        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
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
                noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2],
                                    device=device) if unetG else torch.randn(b_size, 100, 1, device=device)
                noise = noise if etestG else torch.randn(b_size, 249, 1, 1, device=device)
                fake = netG(noise.to(device)).to(device)
                fake_outputs = netD(fake.to(device))
                output = fake_outputs[-1].view(-1)
                # Calculate G's loss based on this output
                errG = criterion_min_max(output, label)
                # Calculate gradients for G
                errG.backward(retain_graph=True)

                if criterion_p2p:
                    errG_p2p = criterion_p2p(fake, data.to(device)) / .01
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errG_p2p.backward(retain_graph=True)
                    errG += errG_p2p
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                # Output training stats
                if i % 50 == 0 or i == iterations:
                    epochTimes.append(time.time() - startTime)
                    # Try one sample
                    noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2],
                                        device=device) if unetG else torch.randn(b_size, 100, 1, device=device)
                    fake = netG(noise.to(device)).to(device)
                    fake_outputs = netD(fake.to(device))
                    output = fake_outputs[-1].view(-1)
                    certainty = output.mean().item()
                    # Print Stats
                    tepoch.set_postfix_str(
                        f"{epochTimes[-1]:.2f}sec: -> [{epoch}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {errD.item():4.4f}, Loss_G: {errG.item():4.4f}, D(x): {D_x:4.4f}, D(G(z)): {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:4.4f}",
                        refresh=True)

                    # If Discriminator has no idea then break
                    if certainty >= 0.45 and certainty <= 0.55 and epoch >= n_epochs:
                        break
                    # Start count again
                    startTime = time.time()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                D_real.append(D_x)
                D_fake.append(D_G_z2)
                D_binary.append((errD_real + errD_fake).item())
                if criterion_perc:
                    D_perc.append(errD_perc.item())

        ################################################
        # GAN's Evaluation: compute_prd_from_embedding #
        ################################################
        precision = []
        recall = []
        eval_data = []
        ref_data = []
        with torch.no_grad():
            netG = netG.cuda()
            for i, data in enumerate(trainloader):
                noise = torch.randn(data.shape[0], trainSet.shape[1], trainSet.shape[2], device=device) if unetG \
                                                            else torch.randn(data.shape[0], 100, 1, device=device)
                noise = noise if etestG \
                            else torch.randn(data.shape[0], 249, 1, 1, device=device)

                # certainty = netD(netG(torch.tensor(sar_data).to(device)))
                # certainty = certainty[-1].view(-1).mean().item()
                g = netG(torch.tensor(noise).to(device))
                g = g.cpu()
                for i_batch in g:
                    eval_data.append(torch.unsqueeze(i_batch, 0))
                # certainty = netD(torch.tensor(sar_data).to(device))
                # certainty = certainty[-1].view(-1).mean().item()
                real = data.cpu()
                for i_batch in torch.tensor(real):
                    ref_data.append(torch.unsqueeze(i_batch, 0))

        #logging.info(f"{len(eval_data)} eval_data: {eval_data[0].shape}, {len(ref_data)} ref_data: {ref_data[0].shape}")
        precision, recall = compute_prd_from_embedding(eval_data, ref_data, netD, num_clusters=2)
        logging.info(f"prd_data: {precision.mean()}, {recall.mean()}\n")
        wandb.log({"precision": precision.mean(),
                   "recall": recall.mean()})
        # PRDplot(prd_data, labels=[0, 1], out_path='./precision_recall_distribution.png')
        out_path = f"./QE/{maskset}maskset/{iter}/precision_recall_distribution{str(training_typo)}.png"
        fig = plt.figure(figsize=(3.5, 3.5), dpi=300)
        plot_handle = fig.add_subplot(111)
        plot_handle.tick_params(axis='both', which='major', labelsize=12)
        plt.plot(recall, precision, 'ro', alpha=0.5, linewidth=3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title("Precision vs Recall distribution")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()
        wandb.log({"Precision Recall Distribution": wandb.Image(out_path)})
    out_path = f"./QE/{maskset}maskset/{iter}/GDLoss{str(training_typo)}.png"
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.plot(D_binary, label='D binary')
    plt.plot(D_perc, label='D perc')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path)
    wandb.log({"GD Loss": wandb.Image(out_path)})
    #plt.savefig(f"../../../scripts/trained_models/{maskset}msks-GDLoss" + str(training_typo) + ".png")

    out_path = f"./QE/{maskset}maskset/{iter}/Daccuracy" + str(training_typo) + ".png"
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracy During Training")
    plt.plot(D_real, label="real")
    plt.plot(D_fake, label="fakes")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path)
    wandb.log({"D Accuracy": wandb.Image(out_path)})
    #plt.savefig(f"../../../scripts/trained_models/{maskset}msks-Daccuracy" + str(training_typo) + ".png")
    # Save the model checkpoint
    torch.save(netG.state_dict(), f"./QE/{maskset}maskset/{iter}/trained_models/Generator{str(training_typo)}.pt")

    return results, netG, netD, training_typo


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def main(*args):
    FLAGS = flags.FLAGS
    warnings.filterwarnings('ignore')
    if isinstance(args[0], str):
        dataset = args[0]
    else:
        #dataset = '../../../scripts/data/etest_maskset_8181x5110.csv'
        dataset = '../../../scripts/data/norm_etest_per_maskset_wrt_limits_8181x5110.csv'
    torch.backends.cudnn.deterministic = True
    cuda = True

    # device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load Dataset
    etestG = True  # Select whether etest or inline dataset will be used. default: True
    #TODO: create the corresponding .csv file for inline dataset and uncomment the following 2 lines.
    #etest_data = sio.loadmat(dataset)['gf_data'] if etestG else np.array(
    #    pd.read_csv('../../../scripts/data/inline_dataset_8181_249.csv'))
    etest_data = pd.read_csv(dataset, index_col=[0])
    logging.info(f"{etest_data.shape}")
    logging.info('############ Train 16 generators for each maskset ############')
    for maskset in range(12,13):
        iter = 0
        if not exists(f"QE/{maskset}maskset/"):
            os.mkdir(f"QE/{maskset}maskset/")
        if not exists(f"QE/{maskset}maskset/{iter}/"):
            os.mkdir(f"QE/{maskset}maskset/{iter}/")
        if not exists(f"QE/{maskset}maskset/{iter}/trained_models/"):
            os.mkdir(f"QE/{maskset}maskset/{iter}/trained_models/")

        trainSet = etest_data[etest_data['maskset'] == maskset]
        trainSet = trainSet.drop(columns=['maskset'])
        trainSet = trainSet.values
        trainSet = (trainSet - trainSet.mean()) / trainSet.std()
        trainSet = trainSet.reshape((-1, trainSet.shape[-1] // 13, 13))
        try:
            depth = FLAGS.depth
            nchan = FLAGS.nchan
            ndf = FLAGS.ndf
            batch_size = FLAGS.batch_size
            n_epochs = FLAGS.n_epochs
            learning_rate = FLAGS.learning_rate
            weight_decay = FLAGS.weight_decay
        except:
            depth = 2
            nchan = 64
            batch_size = 16
            ndf = 64
            n_epochs = 5
            learning_rate = 0.001
            weight_decay = .14
            logging.info("An Error was raised by trying to parse FLAGS.")

        hps = {
            'depth': depth,
            'nchan': nchan,
            # Size of feature maps in discriminator
            'ndf': ndf,
            # batch size during training
            'batch_size': batch_size,
            # Number of workers for dataloader
            'workers': 2,
            # Number of channels in the training images.
            'nc': trainSet.shape[1],
            # Beta1 hyperparam for Adam optimizers
            'beta1': 0.5,
            # Number of GPUs available. Use 0 for CPU mode.
            'ngpu': 1
        }

        wandb.init(project="provenance_attestation", name=f"Original-GAN-{maskset}maskset", entity="chrivasileiou", config=hps)
        unetG = False  # Matters ONLY if 'etest' flag is true.
        resnetG = False
        netG = Unet_Generator(hps) if unetG else Generator(hps)
        netG = ResNetGenerator(hps) if resnetG else netG
        #TODO: need to create the inline dataset to train it per maskset.
        # if wants to generates inline dataset.
        netG = netG if etestG else GeneratorInline(hps)
        # Specify Discriminator
        netD = Discriminator(hps)

        logging.info('\n')
        #logging.info(netG)
        #logging.info(netD)
        summary(netG, input_data=torch.randn(batch_size, 100, 1))
        summary(netD, input_data=torch.randn(batch_size, 393, 13))
        logging.info('############ start training ############')
        results, netG, netD, training_type = train(netG,
                                                   netD,
                                                   trainSet,
                                                   maskset,
                                                   n_epochs=n_epochs,
                                                   batch_size=batch_size,
                                                   lr=learning_rate,
                                                   weight_decay=weight_decay,
                                                   beta1=hps['beta1'],
                                                   workers=hps['workers'],
                                                   ngpu=hps['ngpu'],
                                                   unetG=unetG,
                                                   etestG=etestG,
                                                   device=device,
                                                   iter=iter)
        logging.info('############ end of training ############')
        # run_a_sample(None, unetG, trainSet, hps, training_type, device)

        # Plot 3D PCA
        plot_pca(trainSet, netG, maskset, iter)

        # Remove the closest samples. Expand the distribution quicker
        n_gener_samples = 50
        pop_items = 60
        logging.info(f"trainSet: {trainSet.shape}")
        initial_trainSet = trainSet.copy()
        # if FLAGS.perc:
        # Set up dtype
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                logging.info("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor

        while (len(list(trainSet)) >= 0):
            iter += 1
            if not exists(f"QE/{maskset}maskset/{iter}/"):
                os.mkdir(f"QE/{maskset}maskset/{iter}/")
            if not exists(f"QE/{maskset}maskset/{iter}/trained_models/"):
                os.mkdir(f"QE/{maskset}maskset/{iter}/trained_models/")

            samples = []
            for _ in tqdm(range(n_gener_samples)):
                noise = torch.randn(1, 100, 1)
                samples.append(generate_sample(netG, noise, device))
            samples = np.array(samples)
            # make them 2d arrays
            samples = samples.reshape((len(samples), -1))
            trainSet = trainSet.reshape((len(trainSet), -1))
            logging.info(f"Are the samples unique? {is_unique(samples)}, with mean: {samples.mean()}, and std: {samples.std()}")
            # calculate a point in a multi-dimensional space with the average values across the generated samples.
            average_point_from_gan = samples.mean(axis=0)
            # euclidean distance in a multi-dimensional space.
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
            if len(trainSet) < pop_items:
                break
            trainSet = np.array(trainSet).reshape((-1, 393, 13))
            logging.info(f"trainSet: {trainSet.shape}")

            logging.info('############ start training ############')
            results, netG, netD, training_type = train(netG,
                                                       netD,
                                                       trainSet,
                                                       maskset,
                                                       n_epochs=n_epochs,
                                                       batch_size=batch_size,
                                                       lr=learning_rate,
                                                       weight_decay=weight_decay,
                                                       beta1=hps['beta1'],
                                                       workers=hps['workers'],
                                                       ngpu=hps['ngpu'],
                                                       unetG=unetG,
                                                       etestG=etestG,
                                                       device=device,
                                                       iter=iter)
            logging.info('############ end of training ############')
            plot_pca(trainSet, netG, maskset, iter)
        wandb.finish()

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
