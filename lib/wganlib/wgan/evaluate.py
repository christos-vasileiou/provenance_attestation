from absl import app
import torch.optim as optim
import pandas as pd
import time
from wgan.gan_models import *
from wgan.utils import *
from wgan.metrics import compute_prd_from_embedding
import logging
import warnings
from torchinfo import summary
import wandb
import os
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import auc, PrecisionRecallDisplay, precision_recall_curve, classification_report
from matplotlib import pyplot as plt
import dataframe_image as dfi
os.chdir('../../..')

def train(classifier, trainSet, maskset):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)

    epochTimes = []
    n_epochs = 3
    batch_size = 64
    criterion = nn.BCELoss(reduction='sum').to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, betas=(.5, 0.9), weight_decay=.14)
    # stores results over all epochs
    results = pd.DataFrame(index=list(range(1, n_epochs + 1)),
                           columns=["Train Loss", "Validation Loss", "Test Loss",
                                    "Min Validation Loss"])

    wandb.watch(classifier, log="all", log_freq=5)
    # Create the folder to store files.
    storage_path = f"{os.getcwd()}/scripts/trained_models"
    if not exists(storage_path):
        os.mkdir(storage_path)
    for epoch in range(n_epochs + 1):
        trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=8)
        iterations = len(trainSet) // batch_size
        startTime = time.time()
        for i, data in enumerate(trainloader):

            ## Train with all-real batch
            classifier.zero_grad()
            # Format batch
            real = data.to(device)
            b_size = real.size(0)
            # Forward pass real batch through D
            real_outputs = classifier(real)
            output = real_outputs[-1].view(-1)
            real_label = torch.full((output.shape[0],), 1., dtype=torch.float32, device=device)

            errD_real = criterion(output.float(), real_label.float())
            real_outputs = [r.detach() for r in real_outputs]
            # Calc Gradients in Backward Pass
            errD_real.backward()
            D_x = output.cpu().mean().item()
            optimizer.step()

            # Evaluate the predictions
            output = output.cpu()
            y = torch.where(output > .5, torch.ones(output.shape[0]), torch.zeros(output.shape[0]))
            true_y = torch.count_nonzero(y)
            acc = true_y/y.shape[0]
            precision, recall, _ = precision_recall_curve(real_label.cpu().detach().numpy(), output.cpu().detach().numpy())
            auc_score = auc(recall, precision)

            # Output training stats
            if (i == 0 or i == iterations):
                endTime = time.time()
                epochTimes.append(endTime - startTime)

                # Print Stats
                logging.info(f"{epochTimes[-1]:.2f} sec: Mask-Set: [{maskset}] -> "+
                             f"[{epoch}/{n_epochs}][{i}/{len(trainloader)}]: "+
                             f"Loss: {errD_real.item():.2f}, "+
                             f"Prediction: {D_x:.2f}, "+
                             f"Acc: {acc:.2f}, "+
                             f"AUC: {auc_score:.2f}")
                # Start count again
                startTime = time.time()

            log = {f'Train Loss': errD_real.item(),
                   'Train Prediction': D_x,
                   'Train D Loss Real': acc,
                   'AUC': auc_score}
            wandb.log(log)

        #######################################
        # Evaluation: Evaluate Synthetic Data #
        #######################################
        with torch.no_grad():
            # 1. WGAN Clipping
            #_, ax = plt.subplots(figsize=(7, 7))
            eval_data = []
            x = pd.read_csv(f"{os.getcwd()}/scripts/QE/WGAN-CP-enhanced_dataset-12msk_wrt_limits.csv", index_col=[0]).drop(columns=['maskset']).values
            x = x.reshape((-1, 393, 13))[x.shape[0]//2:]
            trainloader = torch.utils.data.DataLoader(x, batch_size=x.shape[0], shuffle=True, num_workers=8)
            for i, data in enumerate(trainloader):
                d = classifier(data.to(device))[-1]
                d = d.to(torch.device("cpu"))
                for i_batch in d:
                    eval_data.append(torch.squeeze(i_batch))
                del data
            eval_data = torch.tensor(eval_data)
            precision_cp, recall_cp, _ = precision_recall_curve(np.ones_like(eval_data), eval_data.numpy())
            auc_score_cp = auc(recall_cp, precision_cp)

            #display = PrecisionRecallDisplay(precision=precision_cp, recall=recall_cp)
            #display.plot(ax=ax, name=f"WGAN-CP AUC={auc_score_cp:.2f}", marker='o')
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            #plt.savefig(f"{os.getcwd()}/scripts/QE/RecallPrecisionCurve-CP.png")
            #plt.close()
            #eval_data = torch.where(eval_data > .5, torch.ones_like(eval_data), torch.zeros_like(eval_data)).to(int)
            #dfi.export(pd.DataFrame(classification_report(np.ones_like(eval_data), eval_data.numpy(), output_dict=True)).transpose(), f"{os.getcwd()}/scripts/QE/ClassificationReport-CP.png")


            prec, rec = compute_prd_from_embedding(x, trainSet, classifier, num_clusters=2, maskset=12)
            logging.info(f"prd_data: {prec.mean()}, {rec.mean()}")
            #out_path = f"{os.getcwd()}/scripts/QE/precision_recall_distribution.png"
            fig = plt.figure(figsize=(7, 7), dpi=600)
            plot_handle = fig.add_subplot(111)
            plot_handle.tick_params(axis='both', which='major', labelsize=12)
            plt.plot(rec, prec, 'o', alpha=0.5, linewidth=3, label=f"WGAN-CP AUC={auc_score_cp:.2f}")



            # 2. WGAN Gradient Penalty
            #_, ax = plt.subplots(figsize=(7, 7))
            eval_data = []
            x = pd.read_csv(f"{os.getcwd()}/scripts/QE/WGAN-GP-enhanced_dataset-12msk_wrt_limits-final.csv", index_col=[0]).drop(columns=['maskset']).values
            x = x.reshape((-1, 393, 13))[x.shape[0]//2:]
            trainloader = torch.utils.data.DataLoader(x, batch_size=x.shape[0], shuffle=True, num_workers=8)
            for i, data in enumerate(trainloader):
                d = classifier(data.to(device))[-1]
                d = d.to(torch.device("cpu"))
                for i_batch in d:
                    eval_data.append(torch.squeeze(i_batch))
                del data
            eval_data = torch.tensor(eval_data)
            precision_gp, recall_gp, _ = precision_recall_curve(np.ones_like(eval_data), eval_data.numpy())
            auc_score_gp = auc(recall_gp, precision_gp)



            prec, rec = compute_prd_from_embedding(x, trainSet, classifier, num_clusters=2, maskset=12)
            logging.info(f"prd_data: {prec.mean()}, {rec.mean()}")
            out_path = f"{os.getcwd()}/scripts/QE/precision_recall_distribution.png"
            plt.plot(rec, prec, 'o', alpha=0.5, linewidth=3, label=f"WGAN-GP AUC={auc_score_gp:.2f}")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title("Precision vs Recall distribution")
            plt.tight_layout()
            plt.legend(fontsize="medium", loc="lower left")
            plt.savefig(out_path, bbox_inches='tight', dpi=600)
            plt.close()
            return results



            eval_data = torch.where(eval_data > .5, torch.ones_like(eval_data), torch.zeros_like(eval_data)).to(int)
            dfi.export(pd.DataFrame(classification_report(np.ones_like(eval_data), eval_data.numpy(), output_dict=True)).transpose(), f"{os.getcwd()}/scripts/QE/ClassificationReport-GP.png")


        wandb.log({'Test AUC CP': auc_score_cp,
                   'Test AUC': auc_score,
                   'Test AUC GP': auc_score_gp})
        logging.info(f"AUC CP: {auc_score_cp:.2f}, "
                     f"AUC: {auc_score}"
                     f"AUC GP: {auc_score_gp}")

    return results


def main(*args):
    warnings.filterwarnings('ignore')
    if isinstance(args[0], str):
        dataset = args[0]
    else:
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
    printNet = True
    for maskset in range(12, 13):
        iter = 0
        # Get the corresponding data based on the mask set
        trainSet = initial_data[initial_data['maskset'] == maskset]
        trainSet = trainSet.drop(columns=['maskset'])
        trainSet = (trainSet - trainSet.mean()) / trainSet.std()
        trainSet = trainSet.values

        # Reshape and go for training
        trainSet = trainSet.reshape((-1, trainSet.shape[-1] // 13, 13)) if etest else trainSet.reshape(
            (-1, trainSet.shape[-1], 1))
        depth = 2
        nchan = 512
        batch_size = 64
        ndf = 512
        n_epochs = 100
        learning_rate = 0.001
        weight_decay = .14
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
            'ngpu': 1,  # torch.cuda.device_count()
            'weight_decay': weight_decay,
        }

        # Specify Discriminator
        classifier = Discriminator(hps, etest=etest)

        logging.info('\n')
        name = f"Evaluate-Synthetic-Data-{maskset}maskset"
        wandb.init(project="provenance_attestation", name=name, entity="chrivasileiou", config=hps)
        if printNet:
            summary(classifier, input_data=torch.randn(batch_size, 393, 13))
            # logging.info(f"\n{netG}")
            # logging.info(f"\n{classifier}")
            printNet = False

        logging.info(f"trainSet: {trainSet.shape}")
        logging.info('############ start training ############')
        results = train(classifier, trainSet, maskset)
        logging.info('############ end of training ############')

        wandb.finish()


if __name__ == "__main__":
    app.run(main)
















"""# 2. WGAN Clipping (from final version of training)
            #_, ax = plt.subplots(figsize=(7, 7))
            eval_data = []
            x = pd.read_csv(f"{os.getcwd()}/scripts/QE/WGAN-CP-enhanced_dataset-12msk_wrt_limits-final.csv", index_col=[0]).drop(columns=['maskset']).values
            x = x.reshape((-1, 393, 13))[x.shape[0]//2:]
            trainloader = torch.utils.data.DataLoader(x, batch_size=x.shape[0], shuffle=True, num_workers=8)
            for i, data in enumerate(trainloader):
                d = classifier(data.to(device))[-1]
                d = d.to(torch.device("cpu"))
                for i_batch in d:
                    eval_data.append(torch.squeeze(i_batch))
                del data
            eval_data = torch.tensor(eval_data)
            precision_cp_final, recall_cp_final, _ = precision_recall_curve(np.ones_like(eval_data), eval_data)
            auc_score_cp_final = auc(recall_cp_final, precision_cp_final)
            #display = PrecisionRecallDisplay(precision=precision_cp_final, recall=recall_cp_final)
            #display.plot(ax=ax, name=f"WGAN-CP F AUC={auc_score_cp_final:.2f}", marker='o')
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            #plt.savefig(f"{os.getcwd()}/scripts/QE/RecallPrecisionCurve-CP-final.png")
            #plt.close()
            #eval_data = torch.where(eval_data > .5, torch.ones_like(eval_data), torch.zeros_like(eval_data)).to(int)
            #dfi.export(pd.DataFrame(classification_report(np.ones_like(eval_data), eval_data.numpy(), output_dict=True)).transpose(), f"{os.getcwd()}/scripts/QE/ClassificationReport-CP-final.png")


            prec, rec = compute_prd_from_embedding(x, trainSet, classifier, num_clusters=2, maskset=12)
            logging.info(f"prd_data: {prec.mean()}, {rec.mean()}")
            #out_path = f"{os.getcwd()}/scripts/QE/precision_recall_distribution.png"
            #fig = plt.figure(figsize=(7, 7), dpi=600)
            #plot_handle = fig.add_subplot(111)
            #plot_handle.tick_params(axis='both', which='major', labelsize=12)
            #plt.plot(rec, prec, 'ro', alpha=0.5, linewidth=3)
            #plt.xlim([0, 1])
            #plt.ylim([0, 1])
            #plt.xlabel('Recall', fontsize=12)
            #plt.ylabel('Precision', fontsize=12)
            #plt.title("Precision vs Recall distribution")
            #plt.tight_layout()
            #plt.legend(title=f"WGAN-CP AUC={auc_score_cp_final:.2f}", fontsize="x-small", loc="lower left")
            #plt.savefig(out_path, bbox_inches='tight', dpi=600)
            #plt.close()




            # 3. WGAN
            #_, ax = plt.subplots(figsize=(7, 7))
            eval_data = []
            x = pd.read_csv(f"{os.getcwd()}/scripts/QE/WGAN-enhanced_dataset-12msk_wrt_limits.csv", index_col=[0]).drop(columns=['maskset']).values
            x = x.reshape((-1, 393, 13))[x.shape[0]//2:]
            trainloader = torch.utils.data.DataLoader(x, batch_size=x.shape[0], shuffle=True, num_workers=8)
            for i, data in enumerate(trainloader):
                d = classifier(data.to(device))[-1]
                d = d.to(torch.device("cpu"))
                for i_batch in d:
                    eval_data.append(torch.squeeze(i_batch))
                del data
            eval_data = torch.tensor(eval_data)
            precision, recall, _ = precision_recall_curve(np.ones_like(eval_data), eval_data.numpy())
            auc_score = auc(recall, precision)
            #display = PrecisionRecallDisplay(precision=precision_cp, recall=recall_cp)
            #display.plot(ax=ax, name=f"WGAN AUC={auc_score:.2f}", marker='o')
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            #plt.savefig(f"{os.getcwd()}/scripts/QE/RecallPrecisionCurve.png")
            #plt.close()
            #eval_data = torch.where(eval_data > .5, torch.ones_like(eval_data), torch.zeros_like(eval_data)).to(int)
            #dfi.export(pd.DataFrame(classification_report(np.ones_like(eval_data), eval_data.numpy(), output_dict=True)).transpose(), f"{os.getcwd()}/scripts/QE/ClassificationReport.png")


            prec, rec = compute_prd_from_embedding(x, trainSet, classifier, num_clusters=2, maskset=12)
            logging.info(f"prd_data: {prec.mean()}, {rec.mean()}")
            #out_path = f"{os.getcwd()}/scripts/QE/precision_recall_distribution.png"
            #fig = plt.figure(figsize=(7, 7), dpi=600)
            #plot_handle = fig.add_subplot(111)
            #plot_handle.tick_params(axis='both', which='major', labelsize=12)
            #plt.plot(rec, prec, 'o', alpha=0.5, linewidth=3)
            #plt.xlim([0, 1])
            #plt.ylim([0, 1])
            #plt.xlabel('Recall', fontsize=12)
            #plt.ylabel('Precision', fontsize=12)
            #plt.title("Precision vs Recall distribution")
            #plt.tight_layout()
            #plt.legend(title=f"WGAN-CP AUC={auc_score:.2f}", fontsize="x-small", loc="lower left")
            #plt.savefig(out_path, bbox_inches='tight', dpi=600)
            #plt.close()"""