import torch
from absl import flags, app
from absl.flags import FLAGS
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import time
from dim_wgan.gan_models import *
from dim_wgan.utils import *
import logging
from tqdm import tqdm
import warnings
from scipy.stats import wasserstein_distance
from torchinfo import summary
import wandb
import os
os.chdir('../../..')

flags.DEFINE_boolean('perc',
                     False,
                     'set if perceptual loss will be applied.')

flags.DEFINE_boolean('p2p',
                     False,
                     'set if pixel-wise loss will be applied between fakes and ideals.')

flags.DEFINE_string('dataset',
                    f"{os.getcwd()}/scripts/data/gf_4masks_data_format_samples_features_sites.mat",
                    'Set the dataset with real labels.')


def train(netG, netD, trainSet, maskset, validationSet=None, testSet=None, n_epochs=100, batch_size=16, lr=0.001,
          schedFactor=0.1, schedPatience=10, weight_decay=0.14, beta1=0.5, workers=1, ngpu=1, unetG=True, etestG=True,
          device=None, iter=None, LAMBDA_TERM=10, critic_iter=20):
    """Train the neural network.

       Args:
           X (RadarDataSet): training data set in the PyTorch data format
           Y (RadarDataSet): test data set in the PyTorch data format
           n_epochs (int): number of epochs to train for
           learning_rate: starting learning rate (to be decreased over epochs by internal scheduler
    """
    time_zero = time.time()
    # seed and initialization
    torch.cuda.manual_seed_all(123456)
    torch.manual_seed(123456)
    # torch.autograd.set_detect_anomaly(True)

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

    # set up function loss, optimizer, and learning rate scheduler.
    training_typo = get_name(netG)
    training_typo += '-' + 'Etest' if etestG else '-Inline'
    training_typo += '-' + str(iter) if iter else ''
    training_typo += f"-critic{str(critic_iter)}-epochs{n_epochs}"
    logging.info(f"Training type: {training_typo}")

    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9), weight_decay=weight_decay)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9), weight_decay=weight_decay)

    # schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    # schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, 'min', verbose = True, factor = schedFactor, patience = schedPatience)

    # stores results over all epochs
    results = pd.DataFrame(index=list(range(1, n_epochs + 1)),
                           columns=["Train Loss", "Validation Loss", "Test Loss",
                                    "Min Validation Loss"])
    # load a model to train, if specified.
    if FLAGS.load_model:
        netG.load_state_dict(torch.load(f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Generator{str(training_typo)}.pt"))
        netD.load_state_dict((torch.load(f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Discriminator{str(training_typo)}.pt")))
        logging.info(f"{bcolors.OKGREEN}Trained models (G and D) loaded: {maskset} maskset, {str(training_typo)} description{bcolors.ENDC} ")
    else:
        logging.info(f"{bcolors.OKGREEN}Start Training from the scratch: maskset -> {maskset}, description -> {str(training_typo)} {bcolors.ENDC} ")

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    wandb.watch(netG, log="all", log_freq=5)
    wandb.watch(netD, log="all", log_freq=5)
    wdmin = np.inf
    edmin = np.inf
    radius_real = None
    final_G_state_dict = f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Generator{str(training_typo)}-final.pt"
    final_D_state_dict = f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Discriminator{str(training_typo)}-final.pt"
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=workers)
    for epoch in range(n_epochs + 1):
        iterations = len(trainSet) // batch_size
        startTime = time.time()
        #with tqdm(trainloader, unit='batch') as tepoch:
        #    for i, data in enumerate(tepoch):
        #        tepoch.set_description(f"Epoch {epoch}:")
        for i, data in enumerate(trainloader):
            ###################################################
            # (1) Update D network: maximize [D(G(z)) - D(x)] #
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
                real = data.to(device)
                b_size = real.size(0)
                if b_size == 1:
                    continue
                # Forward pass real batch through D
                real_outputs = netD(real)
                output = real_outputs[-1].transpose(1,0)
                # print('OUTPUT D:' ,output.shape)
                # Calculate loss on all-real batch
                # Calculate gradients for D in backward pass
                d_loss_real = output.view(output.shape[0],-1).mean()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2], device=device) \
                    if unetG else torch.randn(b_size, 100, 1, device=device)
                noise = noise if etestG else torch.randn(b_size, 100, 1, device=device)
                #print('NOISE: ', noise.shape)
                # Generate fake image batch with G
                fake = netG(noise)

                # Classify all fake batch with D
                fake_outputs = netD(fake.detach())
                output = fake_outputs[-1].transpose(1,0)
                # Calculate D's loss on the all-fake batch
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                d_loss_fake = output.view(output.shape[0], -1).mean()

                # Train with gradient penalty
                gradient_penalty = calculate_gradient_penalty(real, fake, netD)
                # calculate the Wasserstein loss
                d_loss = d_loss_fake - d_loss_real + LAMBDA_TERM * gradient_penalty
                Wasserstein_loss = d_loss_fake - d_loss_real
                d_loss_fake = d_loss_fake.mean()
                d_loss_real = d_loss_real.mean()
                d_loss.backward(retain_graph=True)
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

            # Since we just updated D, perform another forward pass of all-fake batch through D
            noise = torch.randn(data.shape[0], trainSet.shape[1], trainSet.shape[2],
                                device=device) if unetG else torch.randn(data.shape[0], 100, 1, device=device)
            noise = noise if etestG else torch.randn(b_size, 100, 1,  device=device)
            if b_size == 1:
                continue
            fake = netG(noise.to(device)).to(device)
            fake_outputs = netD(fake.to(device))
            output = fake_outputs[-1].view(-1)
            # Calculate G's loss based on this output
            # Calculate gradients for G
            g_loss = -output.mean()
            g_loss.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if (i % 50 == 0 or i == iterations) and epoch % 50 == 0:
                endTime = time.time()
                epochTimes.append(endTime - startTime)

                # Print Stats
                #tepoch.set_postfix_str(f"Mask-Set: {maskset}, {epochTimes[-1]:.2f}sec: -> [{epoch}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {Wasserstein_loss.item():4.4f}, Loss_G: {g_cost.item():4.4f}",
                #    refresh=True)
                logging.info(f"{bcolors.BOLD}{epochTimes[-1]:.2f} sec{bcolors.ENDC}: Mask-Set: [{bcolors.BOLD}{maskset}{bcolors.ENDC}] -> [{bcolors.OKCYAN}{epoch}{bcolors.ENDC}/{n_epochs}][{i}/{len(trainloader)}] -> Loss_D: {bcolors.OKCYAN}{Wasserstein_loss.item():5.4f}{bcolors.ENDC}, Loss_G: {bcolors.OKCYAN}{g_loss.item():5.4f}{bcolors.ENDC}")
                # Start count again
                startTime = time.time()

            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(Wasserstein_loss.item())
            D_real.append(d_loss_real.item())
            D_fake.append(d_loss_fake.item())
            wandb.log({'Train D Loss Total (fake - real + LAMBDA*gp)': d_loss.item(),
                       'Train D Loss Real': d_loss_real.item(),
                       'Train D Loss Fake': d_loss_fake.item(),
                       'Train Wasserstein Distance (fake-real)': Wasserstein_loss.item(),
                       'Train Gradient Penalty': gradient_penalty.item()
                       })


        ################################################
        # GAN's Evaluation: compute_prd_from_embedding #
        ################################################
        eval_data = []
        ref_data = []
        samples = []
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                noise = torch.randn(data.shape[0], trainSet.shape[1], trainSet.shape[2],
                                    device=device) if unetG else torch.randn(data.shape[0], 100, 1, device=device)
                noise = noise if etestG else torch.randn(data.shape[0], 100, 1, device=device)
                # certainty = netD(netG(torch.tensor(sar_data).to(device)))
                # certainty = certainty[-1].view(-1).mean().item()
                batch = netG(torch.tensor(noise).to(device))
                for i_batch in batch:
                    samples.append(torch.squeeze(i_batch).cpu())
                g = netD(batch)[-1]
                g = g.to(torch.device("cpu"))
                for i_batch in g:
                    eval_data.append(torch.squeeze(i_batch).mean())
                # certainty = netD(torch.tensor(sar_data).to(device))
                # certainty = certainty[-1].view(-1).mean().item()
                d = netD(data.to(device))[-1]
                d = d.to(torch.device("cpu"))
                for i_batch in d:
                    ref_data.append(torch.squeeze(i_batch).mean())
                del data

        eval_data = [t.numpy() for t in eval_data]
        ref_data = [t.numpy() for t in ref_data]
        samples = np.array([t.numpy() for t in samples])
        # Calculate Metrics: Testing
        wdistance = wasserstein_distance(ref_data, eval_data)
        euclidean_distance = np.max(np.sqrt((trainSet.reshape(trainSet.shape[0], -1).mean(0) - samples.reshape(samples.shape[0], -1).mean(0)) ** 2))
        if not radius_real:
            radius_real = np.mean(np.sqrt((trainSet.reshape(trainSet.shape[0], -1) - trainSet.reshape(trainSet.shape[0],-1).mean(0)) ** 2))
        radius_gen = np.mean(np.sqrt((samples.reshape(samples.shape[0], -1) - samples.reshape(samples.shape[0],-1).mean(0)) ** 2))
        wandb.log({'Test Wasserstein Distance': wdistance,
                   'Test Euclidean Distance': euclidean_distance,
                   f'Distance (Radius) of Generated Samples from their Average (Center) R of Real {radius_real}': radius_gen,
                   })
        if epoch % 20 == 0 or epoch == n_epochs-1:
            logging.info(f"{len(eval_data)} eval_data, {len(ref_data)} ref_data")
            logging.info(f"|> Earth Mover's Distance: {bcolors.OKGREEN}{wdistance:.2f}{bcolors.ENDC} <| |> Euclidean's Distance of the Centers of the Distributions (Real-Generated): {bcolors.OKGREEN}{euclidean_distance:.2f}{bcolors.ENDC} <|")

        if np.abs(wdistance) < wdmin and np.abs(euclidean_distance) < edmin:
            wdmin = np.abs(wdistance)
            edmin = np.abs(euclidean_distance)
        #TODO: UN COMMENT
            torch.save(netG.cpu().state_dict(), f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Generator{str(training_typo)}.pt")
            torch.save(netD.cpu().state_dict(), f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Discriminator{str(training_typo)}.pt")
        torch.save(netG.cpu().state_dict(), final_G_state_dict)
        torch.save(netD.cpu().state_dict(), final_D_state_dict)

    # save weights in wandb/online. *if you want to restore them: wandb.restore('filename', run_path='chrivasileiou/provenance_attestation/'id')
    wandb.save(final_G_state_dict)
    wandb.save(final_D_state_dict)

    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracy During Training")
    plt.plot(D_real, label="real")
    plt.plot(D_fake, label="fakes")
    plt.plot(D_losses, label="Wasserstein-D Loss")
    plt.plot(G_losses, label="G Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Daccuracy" + str(training_typo) + ".png")
    # Save the model checkpoint
    torch.save(netG.cpu().state_dict(), final_G_state_dict)
    torch.save(netD.cpu().state_dict(), final_D_state_dict)
    logging.info(f"{bcolors.OKGREEN}Generator Model Saved in the location: {final_G_state_dict}{bcolors.ENDC}")
    logging.info(f"{bcolors.OKGREEN}Discriminator Model Saved in the location: {final_D_state_dict}{bcolors.ENDC}")

    return results, netG, netD, training_typo


def main(*args):
    FLAGS = flags.FLAGS
    warnings.filterwarnings('ignore')
    if isinstance(args[0], str):
        dataset = args[0]
    else:
        #dataset = f"{os.getcwd()}/scripts/data/etest_maskset_8181x5110.csv"
        dataset = f"{os.getcwd()}/scripts/data/norm_etest_per_maskset_wrt_limits_8181x5110.csv"
    torch.backends.cudnn.deterministic = True
    cuda = True

    # device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Number of GPUs in the system: {torch.cuda.device_count()}")
    logging.info(torch.cuda.get_device_properties(device=torch.device("cuda")))
    logging.info(f"Device: {device}")

    # Load Dataset
    etest = True  # Select whether E-test or Inline dataset will be used. default: True
    description = 'E-Test' if etest else 'Inline'
    logging.info(f"{dataset}")
    initial_data = pd.read_csv(dataset, index_col=[0]) if etest else \
        pd.read_csv(f"{os.getcwd()}/scripts/data/inline_maskset_8181x249.csv", index_col=[0])
    logging.info(f"{initial_data.shape}")
    logging.info(f'\n############ Train 16 generators for each maskset. {description} Dataset ############')
    printNet = True
    for maskset in range(12, 13):
        trainSet = initial_data[initial_data['maskset'] == maskset]
        trainSet = trainSet.drop(columns=['maskset'])
        trainSet = (trainSet - trainSet.mean()) / trainSet.std()
        trainSet = trainSet.values
        trainSet = trainSet.reshape((-1, trainSet.shape[-1] // 13, 13)) if etest else trainSet.reshape((-1, trainSet.shape[-1], 1))
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
            logging.info("An Error was raised by trying to parse FLAGS.")
        hps = {
            'depth': depth,
            'nchan': nchan,
            # batch size during training
            'batch_size': batch_size,
            # Number of workers for dataloader
            'workers': 8,
            # Number of channels in the training images.
            'nc': trainSet.shape[1],
            # Size of feature maps in discriminator
            'ndf': ndf,
            # Beta1 hyperparam for Adam optimizers
            'beta1': 0.5,
            # Number of GPUs available. Use 0 for CPU mode.
            'ngpu': 1, #torch.cuda.device_count()
            # term multipyied with gradient penalty loss
            'LAMBDA_TERM': 10,
            # iterations over 1 epoch, the Critic (Discriminator) should be trained.
            'critic_iter': 20
        }

        unetG = False  # Matters ONLY if 'etest' flag is true.
        resnetG = False
        netG = Unet_Generator(hps) if unetG else Generator(hps)
        netG = ResNetGenerator(hps) if resnetG else netG
        netG = netG if etest else GeneratorInline(hps)
        # Specify Discriminator
        netD = Discriminator(hps, wgan=True, etest=etest)
        #netD = InstNDiscriminator(hps, wgan=True, etest=etest)
        #netD = LayerNDiscriminator(hps, wgan=True, etest=etest)

        logging.info('\n')
        if printNet:
            summary(netG, input_data=torch.randn(batch_size, 100, 1))
            summary(netD, input_data=torch.randn(batch_size, 393, 13))
            #logging.info(f"\n{netG}")
            #logging.info(f"\n{netD}")
            printNet = False
        logging.info('############ start training ############')
        iter = 0
        startTime = time.time()
        #TODO: Change to 'provenance_attestation' project
        wandb.init(project="provenance_attestation", name=f'WGAN-GP-BatchNorm-{maskset}maskset', entity="chrivasileiou", config = hps)
        _, netG, netD, training_type = train(netG,
                                               netD,
                                               trainSet,
                                               maskset,
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
                                               etestG=etest,
                                               device=device,
                                               iter=iter)
        endTime = time.time()
        logging.info('############ end of training ############')
        wandb.finish()
        duration = endTime-startTime
        logging.info("Total Training's duration")
        logging.info(print_time(duration))
        logging.info("Duration per epoch:")
        print_time(duration/n_epochs)
        # run_a_sample(None, unetG, trainSet, hps, training_type, device)

        """
        n_gener_samples = 100
        pop_items = 150
        logging.info("trainSet: ", trainSet.shape)
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
            samples = []
            for _ in tqdm(range(n_gener_samples)):
                noise = torch.randn(1, 100, 1)
                samples.append(generate_sample(netG, noise, device, dtype))
            samples = np.array(samples)
            # make them 2d arrays
            samples = samples.reshape((len(samples), -1))
            trainSet = trainSet.reshape((len(trainSet), -1))
            logging.info("Are the samples unique? ", is_unique(samples), " with mean: ", samples.mean(), " and std: ",
                  samples.std())
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
            iter += 1
            trainSet = np.array(trainSet).reshape((-1, 393, 13))
            logging.info("trainSet: ", trainSet.shape)
            del samples
            del distance
            del sort
            logging.info('############ start training ############')
            results, netG, netD, training_type = train(netG,
                                                       netD,
                                                       trainSet,
                                                       maskset,
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
            logging.info('############ end of training ############')
        """

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
