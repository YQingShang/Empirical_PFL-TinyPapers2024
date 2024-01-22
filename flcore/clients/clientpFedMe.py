import numpy as np
import time
import copy
import torch
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.optimizers.fedoptimizer import pFedMeOptimizer
from flcore.clients.clientbase import Client
from ..trainmodel.models import BinaryLogisticRegression


class clientpFedMe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate

        # these parameters are for personalized federated learing.
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = pFedMeOptimizer(
            self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):  # local update
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # K is number of personalized steps
                for i in range(self.K):
                    output = self.model(x)
                    if isinstance(self.model, BinaryLogisticRegression):
                        output = output.squeeze().to(torch.float)
                        y = y.to(torch.float)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # finding aproximate theta
                    self.personalized_params = self.optimizer.step(self.local_params, self.device)

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.update_parameters(self.model, self.local_params)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.update_parameters(self.model, self.personalized_params)
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

        # self.model.cpu()

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auroc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auprc = metrics.average_precision_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auroc, auprc

    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    y = y.to(torch.float)
                loss = self.loss(output, y).item()

                lm = torch.cat([p.data.view(-1) for p in self.local_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.personalized_params], dim=0)
                loss += 0.5 * self.lamda * torch.norm(lm-pm, p=2)

                if isinstance(self.model, BinaryLogisticRegression):
                    train_acc += (torch.sum((output >= 0.5) == y)).item() / y.shape[0]
                else:
                    train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        
        return train_acc, losses, train_num
