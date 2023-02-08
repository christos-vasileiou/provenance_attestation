import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from matplotlib import pyplot
import time
import warnings
import seaborn as sns

def get_shuffle_data(or_data):
    or_col = or_data.columns
    train_data = pd.DataFrame(data=None, index=None, columns=or_col, dtype=None, copy=None)
    test_data = pd.DataFrame(data=None, index=None, columns=or_col, dtype=None, copy=None)
    # Shuffle
    or_data.sample(frac=1).reset_index(drop=True)
    for i in range(16):
        sub_data = or_data.query('mask_data== @i')
        train_sub = sub_data.sample(frac=0.8, random_state=1)
        train_data = train_data.append(train_sub, ignore_index=True)
        test_sub = sub_data.drop(train_sub.index)
        test_data = test_data.append(test_sub, ignore_index=True)
    train_data.to_csv(r"./data/train_data.csv")
    test_data.to_csv(r"./data/test_data.csv")
    return (train_data, test_data)

def normalise(train_data, test_data):
    #need to remove the last two columns
    train_data_x = train_data.iloc[:,:-2]#.values
    train_data_y = train_data.iloc[:,-1]#.values
    test_data_x = test_data.iloc[:,:-2]#.values
    test_data_y = test_data.iloc[:,-1]#.values
    train_mean = np.mean(train_data_x, axis=0)
    train_std = np.std(train_data_x, axis=0)
    train_norm = (train_data_x - train_mean) / train_std
    test_norm = (test_data_x - train_mean) / train_std
    #to check if any column doesnt have na values
    for col in train_norm.columns:
        if train_norm[col].isna().sum() >0:
            #display(col)
            del train_norm[col]
            del test_norm[col]

    return(train_norm,test_norm,train_data_y,test_data_y)

class GFDataset(torch.utils.data.Dataset):
    def __init__(self, norm, y , n_rows=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x_data = torch.from_numpy(np.array(norm.astype('float32'))).to(device)
        self.y_data = torch.from_numpy(np.array(y.astype('int'))).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        trgts = self.y_data[idx]
        sample = {
            'predictors' : preds,
            'targets' : trgts
        }
        return sample

class Vectorize(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Classifier(nn.Module):
    def __init__(self, num_classes=16, lr=1e-3, weight_decay=0., type='cnn', name = 'Classifier'):
        super(Classifier, self).__init__()
        self.name = name
        self.type = type
        self.num_classes = num_classes
        self.model = self.build(type)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(self.ccnmodel.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if type!='fully':
            return
        for i in self.model.modules():
            if not isinstance(i, nn.modules.container.Sequential):
                classname = i.__class__.__name__
                if hasattr(i, "weight"):
                    if classname.find("Conv") != -1:
                        nn.init.xavier_uniform_(i.weight)
                    elif classname.find("BatchNorm2d") != -1:
                        nn.init.normal_(i.weight.data, 1.0, 0.02)
                if hasattr(i, 'bias'):
                    nn.init.zeros_(i.bias)
                    if classname.find("BatchNorm2d") != -1:
                        nn.init.constant_(i.bias.data, 0.0)

    def build(self, type='cnn'):
        if type not in ('cnn', 'fully', 'acnn'):
            raise ValueError("You must specify a type of cnn or fully!")
        if type == 'cnn':
            model = nn.Sequential(
                # input: 1X249
                nn.Conv1d(1, 64*4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(64*4),
                #nn.ReLU(),

                # intermediate: 1X129
                nn.Conv1d(64*4, 64*8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(64*8),
                #nn.ReLU(),

                # intermediate: 1X69
                nn.Conv1d(64*8, 64*16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(64*16),
                #nn.ReLU(),

                # intermediate: 1X34
                nn.Conv1d(64*16, 1, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(1),
                #nn.ReLU(),

                # intermediate: 1X16
                Vectorize(),
                nn.Linear(16, self.num_classes)
                # output: 1X16
            )
        elif type=='fully':
            model = nn.Sequential(
                nn.Linear(249, 512),
                nn.Linear(512, 256),
                nn.Linear(256, self.num_classes)
            )
        elif type=='acnn':
            model = nn.Sequential(
                # input: 1X1X249 channels
                nn.ConvTranspose2d(249, 64*8, kernel_size=2, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64*8),
                # intermediate: 2X2X[64*8]
                nn.ConvTranspose2d(64*8, 64*4, kernel_size=2, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64*4),
                # intermediate: 4X4X[64*4]
                nn.ConvTranspose2d(64*4, 1, kernel_size=2, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                # intermediate: 8X8X[64*8]
                Vectorize(),
                nn.Linear(16, self.num_classes)
                # output: 1X16
            )
        return model

    def classifier(self, x, run_cpu=False):
        if run_cpu == False:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        else:
            self.model.cpu().double()
        if self.type == 'acnn':
            if len(x.shape) == 1: # once evaluation is performed, add an axis as batch_size
                x = x.unsqueeze(dim=0)
            x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        elif self.type == 'cnn':
            if len(x.shape) == 1: # once evaluation is performed, add an axis as batch_size
                x = x.unsqueeze(dim=0)
            x = x.unsqueeze(dim=1)
        if run_cpu:
            out = self.model(x.detach().cpu()).detach().cpu()
        else:
            out = self.model(x)
        #out = F.softmax(out, dim=1) if (len(out.shape) == 2) else F.softmax(out, dim=0)
        return out

    def predict(self, x, run_cpu=True):
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            x = torch.tensor(x.values)
        if run_cpu == False:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            x = x.to(device, dtype=torch.float32)
        else:
            pass
        out = torch.argmax(self.classifier(x, run_cpu=run_cpu), dim=1).detach().numpy()
        return out

    def evaluation(self, dataset):
        # assumes model.eval()
        # granular but slow approach
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the prediction and label lists(tensors)
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        n_correct = 0; n_wrong = 0
        # dataset type: numpy array
        for i in range(len(dataset)):
            with torch.no_grad():
                x = dataset[i]['predictors'].to(device, dtype=torch.float32)
                y = dataset[i]['targets'].to(device)
                outputs = self.classifier(x)
                #.reshape((len(dataset), self.num_classes))  # logits form

            big_idx = torch.argmax(outputs)
            #print('big idx', big_idx, ', y', y.shape, oupt.shape)
            if big_idx == y:
                n_correct += 1
            else:
                n_wrong += 1

            # Append batch prediction results
            predlist=torch.cat([predlist,big_idx.view(-1).cpu()])
            lbllist=torch.cat([lbllist,y.view(-1).cpu()])

        acc = (n_correct * 1.0) / (n_correct + n_wrong)
        cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
        cr = classification_report(lbllist.numpy(), predlist.numpy(), zero_division=1)
        return (acc, cm, cr)

    def fit(self, or_data, n_epochs=30, batch_size=16, num_workers=1):
        warnings.filterwarnings('ignore')
        np.random.seed(1)
        torch.cuda.manual_seed_all(123456)
        torch.manual_seed(123456)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Train Network: \"{self.type}\", on Device: {device}")
        losses = []
        epochTimes = []
        epoch_loss = 0

        results = pd.DataFrame(index = list(range(1, n_epochs+1)), columns = ["Test Accuracy", "Train Accuracy", "Test Loss", "Train Loss"])
        for epoch in range(n_epochs):
            torch.manual_seed(1+epoch)
            train_data, test_data = get_shuffle_data(or_data.copy())
            train_data.to_csv(r"./data/train_data.csv")
            test_data.to_csv(r"./data/test_data.csv")
            del train_data
            del test_data
            train_data = pd.read_csv(r"./data/train_data.csv")
            test_data = pd.read_csv(r"./data/test_data.csv")
            train_norm,test_norm,train_y,test_y = normalise(train_data, test_data)
            train_ds = GFDataset(train_norm, train_y)
            test_ds = GFDataset(test_norm, test_y)
            # TODO: implement reshuffling of the whole dataset
            self.model.train()
            train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            start = time.time()
            for batch_idx, batch in enumerate(train_ldr):
                self.model.zero_grad()
                x = batch['predictors'].to(device, dtype=torch.float32)  # inputs
                y = batch['targets'].to(device, dtype=torch.long)     # shape [b_size,3] (!)
                outputs = self.classifier(x)
                loss = self.criterion(outputs, y)
                epoch_loss += loss.item()  # a sum of averages
                loss.backward()
                self.optimizer.step()
                # Save loss
                losses.append(loss.item())

            test_epoch_loss = 0
            test_ldr = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
            for (batch_idx, batch) in enumerate(test_ldr):
                x = batch['predictors'].to(device, dtype=torch.float32)  # inputs
                y = batch['targets'].to(device, dtype=torch.long)     # shape [b_size,3] (!)
                self.optimizer.zero_grad()
                test_out = self.classifier(x) #.reshape((batch_size, self.num_classes))
                test_loss = self.criterion(test_out, y)
                test_epoch_loss += test_loss.item()

            # evaluate network
            acc_train, cm_train, cr_train = self.evaluation(train_ds)
            acc_test, cm_test, cr_test = self.evaluation(test_ds)
            epochTimes.append(time.time()-start)

            # store results
            results.loc[epoch + 1]["Train Accuracy"] = acc_train
            results.loc[epoch + 1]["Test Accuracy"] = acc_test

            results.loc[epoch + 1]["Train Loss"] = epoch_loss
            results.loc[epoch + 1]["Test Loss"] = test_epoch_loss

            print_acc_train = acc_train*100
            print_acc_test = acc_test*100
            print('[%.1f] sec: [%d/%d] Train Loss: [%4.4f], Train Accuracy: [%2.1f], Test Loss: [%4.4f], Test Accuracy: [%2.1f]' % (
                    epochTimes[-1], epoch, n_epochs, epoch_loss, print_acc_train, test_epoch_loss, print_acc_test))

        acc_train, cm_train, cr_train = self.evaluation(train_ds)  # item-by-item
        acc_test, cm_test, cr_test = self.evaluation(test_ds)

        #print(cm_train.ravel())
        print("\nAccuracy on training data = %0.4f" % (acc_train)) #", tn = %d, fp = %d, fn = %d, tp = %d", tn, fp, fn, tp))
        #print(cm_test.ravel())
        print("Accuracy on test data = %0.4f" % (acc_test)) #", tn = %d, fp = %d, fn = %d, tp = %d", tn, fp, fn, tp))
        print(cr_train)
        print(cr_test)
        # Plot confusion matrix
        sns.heatmap(cm_train, annot=True, annot_kws={'fontsize':6}, fmt='.0f', square=True)
        pyplot.savefig(f"trained_models/ConfusionMatrix-train-{self.type}.png")
        pyplot.close()
        sns.heatmap(cm_test, annot=True, annot_kws={'fontsize':6}, fmt='.0f', square=True)
        pyplot.savefig(f"trained_models/ConfusionMatrix-test-{self.type}.png")
        pyplot.close()

        # Plot Loss values
        pyplot.figure(figsize=(10,5))
        pyplot.plot(results.index, results["Train Loss"])
        pyplot.plot(results.index, results["Test Loss"])
        pyplot.legend(["Train", "Test"])
        pyplot.title('Cross-entropy Loss during Training')
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Cross-Entropy Loss')
        pyplot.savefig(f"./trained_models/CNNClassifierLoss-{self.type}.png")
        pyplot.close()
        # Save the model's checkpoint
        torch.save(self.model.state_dict(), './trained_models/Classifier.pt')
        # Plot Accuracies
        pyplot.figure(figsize = (10, 10))
        pyplot.plot(results.index, results["Train Accuracy"])
        pyplot.plot(results.index, results["Test Accuracy"])
        pyplot.legend(["Train", "Test"])
        pyplot.title('Accuracy during Training')
        pyplot.xlabel('epochs')
        pyplot.ylabel('(%) Accuracy')
        pyplot.savefig(f"./trained_models/CNNClassifier_Accuracy-{self.type}.png")
        pyplot.close()


if __name__ == '__main__':
    or_data = pd.read_csv(r"./data/multiclass_project_dataset_8753X253.csv")
    np.random.seed(1)

    batch_size = 16
    weight_decay = 0.
    n_epochs = 5
    lr = 1e-3
    model = Classifier(lr=lr, weight_decay=weight_decay, type='fully')
    model.fit(or_data, n_epochs=n_epochs, batch_size=batch_size)