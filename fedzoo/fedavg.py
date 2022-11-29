from typing import OrderedDict
import numpy as np
import torch
import arrow
import random

from fedbase import FedBase
from server import CenterServer
from client import Client
from util import NonNegativeClipper

class FedAvg(FedBase):
    def __init__(self, dataset, model, optimizer, optimizer_args, num_client=48, batchsize=4, round = 100
        ):
        super().__init__(dataset=dataset)
        # model 
        self.model = model
        # optimizer 
        self.optimizer = optimizer
        # optimizer hyperparams
        self.optimizer_args = optimizer_args

        # number of client = 52, for each US states
        self.num_client = num_client
        # batchsize = 4, for 4 years' data
        self.batchsize = batchsize
        # communication round
        self.round = round
        # local update epoch = 1
        self.local_epoch = np.ones(self.num_client).astype(int)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("GPU Available, Using Device:", self.device)
        else:
            self.device = torch.device('cpu')
            print("GPU Unavailable, Using Device: ", self.device)

        self.device = 'cpu'
        
        # get train/test data 
        train_data, test_data = self.sampling()
        # convert to tensor
        train_dataloader = [torch.FloatTensor(train_data[i]) for i in range(len(train_data))]
        test_dataloader =  [torch.FloatTensor(test_data[i]) for i in range(len(test_data))]

        # allocate to local client 
        self.client = [FedAvgClient(i, train_dataloader[i], self.device) for i in range(self.num_client)]
        self.datasize = sum([len(i) for i in self.client])
        self.weight = [len(i)/self.datasize for i in self.client]
        
        self.server = FedAvgServer(self.model, test_dataloader, device=self.device)

    def epoch(self):
        print('Training Begin')
        for i in range(self.round):

            print('Current Round {:d}'.format(i+1))
            self.train()
            # val_loss, val_acc = self.val()
            # print('Validation Loss: {:4f}, Validation Accuracy: {:4f}'.format(val_loss, val_acc))
        print('Training End')

        

    def train(self):
        self.num_client = len(self.client)

        for i in self.client:
            i.model = self.server.send_model()

        # train all clients
        selected_client = np.arange(self.num_client)
        
        for i in iter(selected_client):
            self.client[i].client_update(self.local_epoch[i], self.optimizer, self.optimizer_args, self.batchsize)

        self.server.aggregation(self.client, self.weight)


    def val(self):
        val_loss, val_acc = self.server.validation(self.loss_fn)
        return val_loss, val_acc


class FedAvgServer(CenterServer):
    def __init__(self, model, dataloader, device='cpu', minorityloader = None):
        super().__init__(model, dataloader, device='cpu', minorityloader=minorityloader)

    def aggregation(self, client, weight):
        update_state = OrderedDict()

        for i, client_i in enumerate(client):
            local_state = client_i.model.state_dict()

            for key in self.model.state_dict().keys():
                try:
                    update_state[key] += local_state[key] * weight[i] 
                except:
                    update_state[key] = local_state[key] * weight[i]

        self.model.load_state_dict(update_state)

    def validation(self, loss_fn, isRecord = False):
        """
        TODO
        """
        # lossmeter = RecordingMeter()
        # accmeter = RecordingMeter()

        # self.model.eval()
        # self.model.to(self.device)

        # with torch.no_grad():
        #     for x, y in self.dataloader[-1]:
        #         x = x.to(self.device)
        #         y = y.to(self.device)

        #         pred = self.model(x)
        #         loss = loss_fn(pred, y).item()
        #         lossmeter.update(loss, 1)
        #         pred = pred.argmax(dim=1, keepdim=True)
        #         accmeter.update(pred.eq(y.view_as(pred)).sum().item(), y.size(0))

        # return lossmeter.get_avg(), accmeter.get_avg()


class FedAvgClient(Client):
    def client_update(self, local_step, optimizer, optimizer_args, batch_size, print_iter=1, tol=1e-4):
        seed = 500
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # training
        train_llk = self.train(self.model, self.train_data, device=self.device,
                    num_epochs=local_step, optim=optimizer, lr=optimizer_args['lr'], batch_size=batch_size,
                    print_iter=print_iter, tol=tol)


    def train(self, model, train_data, device, num_epochs, optim, lr=1e-4, batch_size=5, print_iter=2, tol=1e-2):
        model.to(device)
        optimizer = optim(model.parameters(), lr=lr)
        clipper1  = NonNegativeClipper()

        best_lglk = -np.inf
        prev_lglk = -np.inf
        no_incre = 0
        converge = 0
        _lr = lr
        n_batches = int(train_data.shape[0] / batch_size) # number of batches

        train_llk = []

        for i in range(num_epochs):
            try:
                epoch_llk_loss = 0
                optimizer.zero_grad()

                for b in range(n_batches):
                    idx = np.arange(batch_size * b, batch_size * (b + 1))
                    data = train_data[idx]
                    loglik = model(data.to(device))
                    loss = - loglik

                    loss.backward()
                    optimizer.step()
                    model.apply(clipper1)
                
                    epoch_llk_loss += loglik.item()
                
                event_num = (train_data[..., 0] > 0).sum()
                event_llk = epoch_llk_loss / event_num

                if (i+1) % print_iter == 0:
                    train_llk.append(event_llk)
                    print("[%s] Epoch: %d\tTrain Loglik: %.3e\t stag: %d converge: %d" % (arrow.now(), i, event_llk, no_incre, converge))
                    
                if event_llk > best_lglk:
                    best_lglk = event_llk
                    no_incre = 0
                else:
                    no_incre += 1

                if no_incre == 10:
                    print("Learning rate decrease!")
                    _lr = _lr / 10
                    optimizer = optim(model.parameters(), lr=_lr)
                    no_incre = 0

                if np.abs(event_llk - prev_lglk) > tol:
                    converge = 0
                else:
                    converge += 1

                prev_lglk = event_llk
                
                if converge == 50:
                    return train_llk

            except KeyboardInterrupt:
                break
        
        return train_llk
