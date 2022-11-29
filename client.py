class Client(object):
    def __init__(self, client_id, train_data, device='cpu'):
        self.client_id = client_id
        self.train_data = train_data
        self.device = device
        self._model = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self, local_step, loss_fn, optimizer, optimizer_args):
        raise NotImplementedError

    def __len__(self):
        return len(self.train_data)

    def loss_estimator(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError