import copy

class CenterServer(object):
    def __init__(self, model, dataloader, device='cpu', minorityloader = None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.minorityloader = minorityloader

    def aggregation(self):
        raise NotImplementedError

    def send_model(self):
        return copy.deepcopy(self.model)

    def validation(self):
        raise NotImplementedError

    def record(self):
        raise NotImplementedError
