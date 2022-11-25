import numpy as np 
from src.dataset import FMNISTDataset
import json

class FedBase(object):
    def __init__(self, dataset=None):
        self.dataset = dataset

    def sampling(self, num_client, iid=True):
        if self.dataset is None:
            raise Exception('Please Specify the Dataset')

        # FMNIST Dataset Sampling
        if self.dataset == 'FMNIST':
            with open('./dataset/FMNIST/FMNIST_train.json') as f:
                train = json.load(f)
            with open('./dataset/FMNIST/FMNIST_test.json') as f:
                test = json.load(f)

            local_dataset = []
            num_samples = np.random.randint(low=150, high=450, size=num_client)

            if iid is True:
                train_x, train_y = [], []
                for i in train['users']:
                    train_x += train['user_data'][i]['x']
                    train_y += train['user_data'][i]['y']

                for i,n in zip(num_samples, range(num_client)):
                    selected_index = np.random.choice(range(len(train_y)), size=i, replace=False)
                    local_dataset.append(FMNISTDataset(np.array(train_x)[selected_index].reshape(-1,1,28,28), np.array(train_y)[selected_index], n))

            if iid is False:
                cat = ['0:T-shirt(top)', '2:pullover', '6:shirt']
                for i,n in zip(num_samples, range(num_client)):
                    index = int(n / int(num_client/3))
                    train_x = train['user_data'][cat[index]]['x']
                    train_y = train['user_data'][cat[index]]['y']

                    selected_index = np.random.choice(range(len(train_y)), size=i, replace=False)
                    local_dataset.append(FMNISTDataset(np.array(train_x)[selected_index].reshape(-1,1,28,28), np.array(train_y)[selected_index], n))

            val_x, val_y = [], []
            val_dataset = []
            for i in test['users']:
                temp_x = test['user_data'][i]['x']
                temp_y = test['user_data'][i]['y']
                val_dataset.append(FMNISTDataset(np.array(temp_x).reshape(-1,1,28,28), np.array(temp_y), client_id=-1))
                val_x += temp_x
                val_y += temp_y

            val_dataset.append(FMNISTDataset(np.array(val_x).reshape(-1,1,28,28), np.array(val_y), client_id=-1))
            
            return local_dataset, val_dataset

        """
        TODO:

        Implement your own data allocation strategy here
        """


    def train(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def epoch(self):
        raise NotImplementedError