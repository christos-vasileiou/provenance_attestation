from absl import flags, app
from absl.flags import FLAGS
import torch.optim as optim
import pandas as pd
import time
from wgan.gan_models import *
from wgan.utils import *
import logging
from tqdm import tqdm
import warnings
from scipy.stats import wasserstein_distance
from torchinfo import summary
import wandb
import os
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
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

flags.DEFINE_boolean('hypercube',
                     False,
                     'set if the hyper cube reduction will be applied in multiple training')

flags.DEFINE_integer('critic_iter',
                     20,
                     'How many times Critic (Discriminator) will be trained more times than G')

flags.DEFINE_integer('type',
                     2,
                     'Define the type of the Discriminator (Critic). Either WGAN-Clipping=1 or WGAN-GP=2')

def train(netG, netD, trainSet, maskset, n_epochs=100, batch_size=16, lr=0.001,
          weight_decay=0.14, beta1=0.5, workers=1, ngpu=1, unetG=True, etestG=True,
          device=None, iter=None, LAMBDA_TERM=10, critic_iter=20, first_time=True):
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
    weight_clipping_limit = 1.
    gp = ' + LAMBDA*gp' if FLAGS.type == 2 else ''

    # set up function loss, optimizer, and learning rate scheduler.
    training_typo = get_name(netG)
    training_typo += '-' + 'Etest' if etestG else '-Inline'
    training_typo += '-' + str(iter) if iter else ''
    training_typo += f"-critic{str(critic_iter)}-epochs{n_epochs}"
    logging.info(f"Training type: {training_typo}")

    # Setup Adam optimizers for both G and D
    if FLAGS.type == 1:
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
    if FLAGS.load_model and first_time:
        #filepath = f"{os.getcwd()}/scripts/trained_models/InstanceNorm/{maskset}msks-Generator-Generator-BCELoss-sum-Etest-critic20-epochs4000-final.pt"
        filepath = f"{os.getcwd()}/scripts/trained_models/InstanceNorm/{FLAGS.load_model}"
        netG.load_state_dict(torch.load(filepath))
        #netD.load_state_dict((torch.load(f"{os.getcwd()}/scripts/trained_models/{maskset}msks-Discriminator{str(training_typo)}.pt")))
        logging.info(f"{bcolors.OKGREEN}Trained model G loaded from the path: {filepath}{bcolors.ENDC}, for the maskset: {maskset}")
    else:
        logging.info(f"{bcolors.OKGREEN}Start Training from the scratch: maskset -> {maskset}, description -> {str(training_typo)} {bcolors.ENDC} ")

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    wandb.watch(netG, log="all", log_freq=5)
    wandb.watch(netD, log="all", log_freq=5)
    #wandb.watch(netD, log="all", log_freq=5)
    wd = np.inf if FLAGS.type == 2 else 0
    edmin = np.inf
    # Create the folder to store files.
    storage_path = f"{os.getcwd()}/scripts/trained_models/WGAN-CP/{iter}" if FLAGS.type == 1 else f"{os.getcwd()}/scripts/trained_models/WGAN-GP/{iter}"
    if not exists(storage_path) and iter is not None:
        os.mkdir(storage_path)
    final_G_state_dict = f"{storage_path}/{maskset}msks-Generator{str(training_typo)}-final.pt"
    final_D_state_dict = f"{storage_path}/{maskset}msks-Discriminator{str(training_typo)}-final.pt"
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=workers)
    for epoch in range(n_epochs + 1):
        iterations = len(trainSet) // batch_size
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
                real = data.to(device)
                b_size = real.size(0)
                if FLAGS.type == 1:
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
                noise = torch.randn(b_size, trainSet.shape[1], trainSet.shape[2], device=device) \
                    if unetG else torch.randn(b_size, 100, 1, device=device)
                noise = noise if etestG else torch.randn(b_size, 100, 1, device=device)
                #print('NOISE: ', noise.shape)
                fake = netG(noise)

                # Classify all fake batch with D
                fake_outputs = netD(fake.detach())
                output = fake_outputs[-1].view(-1)
                # Calculate D's loss on the all-fake batch
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                d_loss_fake = output.mean()

                # Train with gradient penalty
                # calculate the Wasserstein loss
                d_loss = d_loss_fake - d_loss_real
                if FLAGS.type == 2:
                    gradient_penalty = calculate_gradient_penalty(real, fake, netD)
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
            if (i % 50 == 0 or i == iterations) and epoch % 10 == 0:
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
            log = {f'Train D Loss Total (fake - real{gp})': d_loss.item(),
                       'Train D Loss Real': d_loss_real.item(),
                       'Train D Loss Fake': d_loss_fake.item(),
                       'Train Wasserstein Distance (fake-real)': Wasserstein_loss.item()}
            if FLAGS.type == 2:
                log.update({'Train Gradient Penalty': gradient_penalty.item()})
            wandb.log(log)

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
                noise = torch.randn(data.shape[0], trainSet.shape[1], trainSet.shape[2],
                                    device=device) if unetG else torch.randn(data.shape[0], 100, 1, device=device)
                noise = noise if etestG else torch.randn(data.shape[0], 100, 1, device=device)

                batch = netG(torch.tensor(noise).to(device))
                for i_batch in batch:
                    samples.append(torch.squeeze(i_batch).cpu())
                g = netD(batch)[-1]
                g = g.to(torch.device("cpu"))
                for i_batch in g:
                    eval_data.append(torch.squeeze(i_batch))
                d = netD(data.to(device))[-1]
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
        euclidean_distance = np.max(np.sqrt((trainSet.reshape(trainSet.shape[0], -1).mean(0) - samples.reshape(samples.shape[0], -1).mean(0)) ** 2))
        wandb.log({'Test Wasserstein Distance': wdistance,
                   'Test Euclidean Distance': euclidean_distance,
                   'Standard Deviation of Real': trainSet.std(),
                   'Standard Deviation of Synthetic': samples.std()})
        if epoch % 20 == 0 or epoch == n_epochs-1:
            logging.info(f"{len(eval_data)} eval_data, {len(ref_data)} ref_data")
            logging.info(f"Earth Mover's Distance: {bcolors.OKGREEN}{wdistance:.2f}{bcolors.ENDC}, Euclidean's Distance of the Centers of the Distributions (Real-Generated): {bcolors.OKGREEN}{euclidean_distance:.2f}{bcolors.ENDC}, Synth STD: {samples.std()}")

        if np.abs(euclidean_distance) < edmin:
            edmin = np.abs(euclidean_distance)

            if FLAGS.type == 1 and np.abs(euclidean_distance) < 2. and samples.std() > trainSet.std()-.1 and samples.std() < trainSet.std()+.6:
                torch.save(netG.cpu().state_dict(), f"{storage_path}/{maskset}msks-Generator{str(training_typo)}.pt")
                torch.save(netD.cpu().state_dict(), f"{storage_path}/{maskset}msks-Discriminator{str(training_typo)}.pt")
            elif FLAGS.type == 2 and np.abs(euclidean_distance) < 2. and samples.std() > trainSet.std()-.1 and samples.std() < trainSet.std()+.6:
                torch.save(netG.cpu().state_dict(), f"{storage_path}/{maskset}msks-Generator{str(training_typo)}.pt")
                torch.save(netD.cpu().state_dict(), f"{storage_path}/{maskset}msks-Discriminator{str(training_typo)}.pt")
        torch.save(netG.cpu().state_dict(), final_G_state_dict)
        torch.save(netD.cpu().state_dict(), final_D_state_dict)
        if samples.std() < 0.5:
            break

    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracy During Training")
    plt.plot(D_real, label="real")
    plt.plot(D_fake, label="fakes")
    plt.plot(D_losses, label="Wasserstein-D Loss")
    plt.plot(G_losses, label="G Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    out_filepath = f"{storage_path}/{maskset}msks-GDLoss" + str(training_typo) + ".png"
    plt.savefig(out_filepath)
    wandb.log({'GDLoss': wandb.Image(out_filepath)})
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
        iter = 0
        # Get the corresponding data based on the mask set
        trainSet = initial_data[initial_data['maskset'] == maskset]
        trainSet = trainSet.drop(columns=['maskset'])
        trainSet = (trainSet - trainSet.mean()) / trainSet.std()
        trainSet = trainSet.values

        # Reshape and go for training
        trainSet = trainSet.reshape((-1, trainSet.shape[-1] // 13, 13)) if etest else trainSet.reshape((-1, trainSet.shape[-1], 1))
        try:
            depth = FLAGS.depth
            nchan = FLAGS.nchan
            batch_size = FLAGS.batch_size
            ndf = FLAGS.ndf
            n_epochs = FLAGS.n_epochs
            learning_rate = FLAGS.learning_rate
            weight_decay = FLAGS.weight_decay
            critic_iter = FLAGS.critic_iter
        except:
            depth = 2
            nchan = 64
            batch_size = 16
            ndf = 64
            n_epochs = 5
            learning_rate = 0.001
            weight_decay = .14
            critic_iter = 20
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
            'lr': learning_rate,
            'LAMBDA_TERM': 10,
            'weight_decay': weight_decay,
            'device': device,
            'iter': iter,
            # iterations over 1 epoch, the Critic (Discriminator) should be trained.
            'critic_iter': critic_iter
        }

        unetG = False  # Matters ONLY if 'etest' flag is true.
        resnetG = False
        # Specify Generator
        netG = Unet_Generator(hps) if unetG else Generator(hps)
        netG = ResNetGenerator(hps) if resnetG else netG
        netG = netG if etest else GeneratorInline(hps)
        # Specify Discriminator
        # If select WGAN-Clipping, get a Discriminator with Batch Normalization
        # otherwise if WGAN-GP, get a Discriminator with Instance Normalization
        netD = Discriminator(hps, wgan=True, etest=etest) if FLAGS.type == 1 else InstNDiscriminator(hps, wgan=True, etest=etest)

        #netD = LayerNDiscriminator(hps, wgan=True, etest=etest)

        logging.info('\n')
        name = f"WGAN-Clipping-{maskset}maskset" if FLAGS.type == 1 else f"WGAN-GP-{maskset}maskset"
        wandb.init(project="provenance_attestation", name=name, entity="chrivasileiou", config=hps)
        if printNet:
            summary(netG, input_data=torch.randn(batch_size, 100, 1))
            summary(netD, input_data=torch.randn(batch_size, 393, 13))
            #logging.info(f"\n{netG}")
            #logging.info(f"\n{netD}")
            printNet = False

        logging.info(f"trainSet: {trainSet.shape}")
        logging.info('############ start training ############')
        startTime = time.time()
        _, netG, netD, training_type = train(netG,
                                               netD,
                                               trainSet,
                                               maskset,
                                               n_epochs=n_epochs,
                                               batch_size=hps['batch_size'],
                                               lr=hps['lr'],
                                               weight_decay=hps['weight_decay'],
                                               beta1=hps['beta1'],
                                               workers=hps['workers'],
                                               ngpu=hps['ngpu'],
                                               unetG=unetG,
                                               etestG=etest,
                                               device=hps['device'],
                                               iter=hps['iter'],
                                               critic_iter=hps['critic_iter'],
                                               first_time=True)
        endTime = time.time()
        logging.info('############ end of training ############')
        duration = endTime-startTime
        logging.info("Total Training's duration")
        logging.info(print_time(duration))
        logging.info("Duration per epoch:")
        print_time(duration/n_epochs)

        n_gener_samples = 50
        pop_items = hps['batch_size']
        # Set up dtype
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                logging.info("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor
        
        while FLAGS.hypercube and (len(list(trainSet)) > 0):
            n_epochs += 150
            samples = []
            for _ in tqdm(range(n_gener_samples)):
                noise = torch.randn(1, 100, 1)
                samples.append(generate_sample(netG, noise, device, dtype))
            samples = np.array(samples)
            # make them 2d arrays
            samples = samples.reshape((len(samples), -1))
            trainSet = trainSet.reshape((len(trainSet), -1))
            logging.info(f"Are the samples unique? {is_unique(samples)}, with mean: {samples.mean()}, and std: {samples.std()}")
            # calculate a point in a multi-dimensional space with the average values across the generated samples.
            trainSet = pop_n_items(trainSet, samples, pop_items)
            if len(trainSet) < pop_items:
                break
            iter += 1
            trainSet = np.array(trainSet).reshape((-1, 393, 13))
            logging.info(f"trainSet: {trainSet.shape}, {type(trainSet[0])}")
            del samples

            startTime = time.time()
            logging.info('############ start training ############')
            results, netG, netD, training_type = train(netG,
                                                       netD,
                                                       trainSet,
                                                       maskset,
                                                       n_epochs=n_epochs,
                                                       batch_size=hps['batch_size'],
                                                       lr=hps['lr'],
                                                       weight_decay=hps['weight_decay'],
                                                       beta1=hps['beta1'],
                                                       workers=hps['workers'],
                                                       ngpu=hps['ngpu'],
                                                       unetG=unetG,
                                                       device=hps['device'],
                                                       iter=hps['iter'],
                                                       critic_iter=hps['critic_iter'],
                                                       first_time=False)
            logging.info('############ end of training ############')
            endTime = time.time()
            duration = endTime-startTime
            logging.info("Total Training's duration")
            logging.info(print_time(duration))
            logging.info("Duration per epoch:")
            print_time(duration/n_epochs)
        wandb.finish()


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
