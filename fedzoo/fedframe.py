from typing import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.fedbase import FedBase
from src.server import CenterServer
from src.client import Client
from src.util import RecordingMeter

class FedAvg(FedBase):
    def __init__(self, dataset, model, optimizer, optimizer_args,
        num_client=100, batchsize=10, round = 100, fraction = 0.1, init_epoch = 100, iid = False
        ):
        super().__init__(dataset=dataset)
        # self.model = model
        # self.optimizer = optimizer
        # self.optimizer_args = optimizer_args

        # self.num_client = num_client
        # self.batchsize = batchsize
        # self.round = round
        # self.fraction = fraction
        # self.init_epoch = init_epoch
        # self.loss_fn = CrossEntropyLoss()
        # self.iid = iid
        # self.local_epoch = [init_epoch for i in range(self.num_client)]

        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        #     print("GPU Available, Using Device:", self.device)
        # else:
        #     self.device = torch.device('cpu')
        #     print("GPU Unavailable, Using Device: ", self.device)
        
        # local_dataset, val_dataset = self.sampling(self.num_client, iid=self.iid)
        # local_dataloader = [DataLoader(dataset, num_workers=0, batch_size=self.batchsize, shuffle=True, drop_last=True) for dataset in local_dataset]
        # val_dataloader =  [DataLoader(dataset, num_workers=0, batch_size=self.batchsize, shuffle=True, drop_last=True) for dataset in val_dataset]

        # self.client = [FedFrameClient(i, local_dataloader[i], self.device) for i in range(self.num_client)]
        # self.datasize = sum([len(i) for i in self.client])
        # self.weight = [len(i)/self.datasize for i in self.client]
        
        # self.server = FedFrameServer(self.model, val_dataloader, device=self.device)

        """
        TODO:

        Do your own initialization here
        """

    def epoch(self):
        print('Training Begin')
        for i in range(self.round):

            """
            TODO:

            Implement own interfence of train/val logger
            """

            print('Current Round {:d}'.format(i+1))
            self.train()

        print('Training End')

        

    def train(self):
        self.num_client = len(self.client)

        for i in self.client:
            i.model = self.server.send_model()


        # N = max(int(self.num_client * self.fraction), 1)
        # selected_client = np.random.randint(0, self.num_client, N)

        """
        TODO:

        Implement your own client selection strategy
        """
        
        for i in iter(selected_client):
            self.client[i].client_update(self.local_epoch[i], self.loss_fn, self.optimizer, self.optimizer_args)

        self.server.aggregation(self.client, self.weight, selected_client)


    def val(self):
        val_loss, val_acc = self.server.validation(self.loss_fn)
        return val_loss, val_acc


class FedFrameServer(CenterServer):
    def __init__(self, model, dataloader, device='cpu', minorityloader = None):
        super().__init__(model, dataloader, device='cpu', minorityloader=minorityloader)

    def aggregation(self, client, weight, seleccted_client):
        # update_state = OrderedDict()

        # for i, client_i in enumerate(client):
        #     local_state = client_i.model.state_dict()

        #     for key in self.model.state_dict().keys():
        #         try:
        #             update_state[key] += local_state[key] * weight[i] 
        #         except:
        #             update_state[key] = local_state[key] * weight[i]

        # self.model.load_state_dict(update_state)

        """
        TODO:

        Implement your own aggregation method here
        """

    def validation(self, loss_fn, isRecord = False):
        lossmeter = RecordingMeter()
        accmeter = RecordingMeter()

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for x, y in self.dataloader[-1]:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = loss_fn(pred, y).item()
                lossmeter.update(loss, 1)
                pred = pred.argmax(dim=1, keepdim=True)
                accmeter.update(pred.eq(y.view_as(pred)).sum().item(), y.size(0))

        return lossmeter.get_avg(), accmeter.get_avg()



class FedFrameClient(Client):
    def client_update(self, local_step, loss_fn, optimizer, optimizer_args):
        self.model.train()
        self.model.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)

        for i in range(local_step):
            for x, y in self.dataloader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

        self.model.to('cpu')

