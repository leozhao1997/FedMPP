import torch
import numpy as np

from config import get_config
from fedzoo.fedavg import FedAvg
from model.markedpp import MultivariateExponentialHawkes
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU Available, Using Device:", device)
else:
    device = torch.device('cpu')
    print("GPU Unavailable, Using Device:", device)
    


device = torch.device('cpu')    


def main():

    T          = np.array([0., 50.])
    data_dim   = 2
    n_class = 12

    alphas = np.random.uniform(low=0.0, high=1.0, size=(n_class, n_class))
    beta   = np.random.uniform(low=0.0, high=1.0, size=(n_class))

    model = MultivariateExponentialHawkes(T=T, mu=0.01*np.ones(n_class), alphas=alphas, beta=beta,
                                           data_dim=data_dim, device=device)
    optimizer = optim.Adadelta
    optimizer_args = {'lr':1e-2}


    fed = FedAvg(
        dataset='Outbreak', model=model, optimizer=optimizer, optimizer_args=optimizer_args,
    )

    fed.epoch()




if __name__ == '__main__':
    main()
