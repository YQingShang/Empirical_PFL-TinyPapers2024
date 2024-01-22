import numpy as np
import torch
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from ..trainmodel.models import BinaryLogisticRegression
from sklearn import metrics
from sklearn.preprocessing import label_binarize

class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self, new_local_epochs):
        trainloader = self.load_train_data(self.batch_size*2)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = new_local_epochs # self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        print("Client size: {} uses local epochs: {}".format(len(trainloader.dataset), max_local_epochs))
        for step in range(max_local_epochs):  # local update
            for X, Y in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))
                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    y = y.to(torch.float)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    y = y.to(torch.float)
                loss = self.loss(output, y)
                #print(loss)
                loss.backward()

                 # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)
            #print('\nlocal epoch step: {}'.format(step))
            #test_acc, test_num, auroc, auprc=self.test_metrics1()
            #print('\nAUROC is: {}'.format(auroc))

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()

        (x, y) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)
        if isinstance(self.model, BinaryLogisticRegression):
            output = output.squeeze().to(torch.float)
            y = y.to(torch.float)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.model.cpu()


    def train_metrics(self, model=None):
        trainloader = self.load_train_data(self.batch_size*2)
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        for X, Y in trainloader:
            # step 1
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][:self.batch_size].to(self.device)
                x[1] = X[1][:self.batch_size]
            else:
                x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            if isinstance(self.model, BinaryLogisticRegression):
                output = output.squeeze().to(torch.float)
                y = y.to(torch.float)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            # step 2
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][self.batch_size:].to(self.device)
                x[1] = X[1][self.batch_size:]
            else:
                x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            if isinstance(self.model, BinaryLogisticRegression):
                output = output.squeeze().to(torch.float)
                y = y.to(torch.float)
            loss1 = self.loss(output, y)

            train_num += y.shape[0]
            losses += loss1.item() * y.shape[0]

        return losses, train_num

    def train_one_epoch(self):
        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            output = self.model(x)
            if isinstance(self.model, BinaryLogisticRegression):
                output = output.squeeze().to(torch.float)
                y = y.to(torch.float)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def test_metrics1(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if isinstance(self.model, BinaryLogisticRegression):
                    test_acc += (torch.sum((output >= 0.5) == y)).item() / y.shape[0]
                else:
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                # print('output:\n', output)
                if isinstance(self.model, BinaryLogisticRegression):
                    y_true.append(y.detach().cpu().numpy())
                else:
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)
                # print('truth:\n', y.detach().cpu().numpy())

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auroc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auprc = metrics.average_precision_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auroc, auprc