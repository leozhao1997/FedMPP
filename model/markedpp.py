import torch

# kernel 
class Multivariate_Exponential_Kernel(torch.nn.Module):
    """
    Exponential Decaying Kernel
    """
    def __init__(self, alphas, beta, device):
        """
        Arg:
        - alphas: influence coefficient matrix, numpy array                     [ n_class, n_class ]
        - beta: temporal decaying parameter for historical influence, numpy array        [ n_class ]
        - device: cpu or cuda
        """
        super(Multivariate_Exponential_Kernel, self).__init__()

        # configuration
        self._alphas     = torch.nn.Parameter(torch.tensor(alphas))
        self._beta       = torch.nn.Parameter(torch.tensor(beta))
        self.n_class     = alphas.shape[0]

    def init_paras(self):
        """
        initialize model parameters
        """

        self._alphas.data = torch.empty(torch.empty(self.n_class, self.n_class).uniform_(0,to=self.init_std)).to(self._alphas.device)
        self._beta.data = torch.empty(torch.empty(self.n_class).uniform_(0,to=self.init_std)).to(self._beta.device)
    
    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at index x and y 
        - x: the current input (t, type) [ batch_size, data_dim = 2 ]
        - y: the history input (t, type) [ batch_size, data_dim = 2 ]
        """

        batch_size   = x.shape[0]
        alphas_hist  = self._alphas[x[:, 1].long(), y[:, 1].long()]
        beta_hist    = self._beta[y[:, 1].long()]

        mask         = x[:, 0] > 0
        tds          = (x[:, 0] - y[:, 0]) * mask

        return alphas_hist * beta_hist * torch.exp(- beta_hist * tds) * mask    # [ batch_size ]


# base point process 
class BasePointProcess(torch.nn.Module):
    """
    PyTorch Module for Multivariate Point Process with diffusion kernel
    """

    def __init__(self, T, mu, data_dim, device):
        """
        data dim = 8

        Args:
        """
        super(BasePointProcess, self).__init__()
        self.data_dim      = data_dim
        self.T             = T # time horizon. e.g. (0, 1)
        self.device        = device
        self.n_class       = len(mu)
        
        # parameters
        self._mu = torch.nn.Parameter(torch.tensor(mu), requires_grad=False)
        # # pre-compute
        # self.data_transformation()

    def cond_lambda(self, xi, hti):
        """
        return conditional intensity given x
        Args:
        - xi:   current i-th point       [ batch_size, data_dim ]
        - hti:  history points before ti [ batch_size, seq_len, data_dim ]
        Return:
        - lami: i-th lambda              [ batch_size ]
        """
        batch_size, seq_len, _ = hti.shape
        mask_all = (hti[:, :, 0] > 0) * (xi[:, 0] > 0)[:, None]
        # if length of the history is zero
        if seq_len == 0:
            return self._mu[xi[:, 1].long()]                                    # [ batch_size ]

        xi2  = xi.unsqueeze(-2).repeat(1, seq_len, 1)                           # [ batch_size, seq_len, data_dim ]
        K    = self.kernel(xi2.reshape(-1, self.data_dim),
                           hti.reshape(-1, self.data_dim)).reshape(batch_size, seq_len)                                           
                                                                                # [ batch_size, seq_len ]
        lami = (K * mask_all).sum(-1) + self._mu[xi[:, 1].long()]                                 
                                                                                # [ batch_size ]
        return lami             # [ batch_size ]

    def log_likelihood(self, X):
        """
        return log-likelihood given sequence X
        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        scalar
        """
        
        raise NotImplementedError()

    def forward(self, X):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)



# marked point process 
class MultivariateExponentialHawkes(BasePointProcess):
    """
    PyTorch Module for Multivariate Temporal Point Process
    """
    def __init__(self, T, mu, alphas, beta, data_dim, device):
        """
        data dim = 5(time, x, y, type)

        Args:
        """
        super(MultivariateExponentialHawkes, self).__init__(T, mu, data_dim, device)
        
        # kernel
        self.kernel = Multivariate_Exponential_Kernel(alphas, beta, device)
        # # pre-compute
        # self.data_transformation()

    def log_likelihood(self, X):
        """
        return log-likelihood given sequence X
        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        scalar
        """

        batch_size, seq_len, _ = X.shape
        ts       = X[:, :, 0].clone()
        ms       = X[:, :, 1].clone().long()
        mask     = ts > 0

        lams     = [
            self.cond_lambda(X[:, i, :].clone(), X[:, :i, :].clone())
            for i in range(seq_len) ]
        lams     = torch.stack(lams, dim=0).T                                   # [ batch_size, seq_len ]
        ## log-likelihood
        mask     = ts > 0                                                       # [ batch_size, seq_len ]
        sumlog   = (torch.log(lams + 1e-8) * mask).sum()                        # [ seq_len ]

        baserate = torch.sum(self._mu * (self.T[1] - self.T[0]))

        temp_int = 1 - torch.exp(-self.kernel._beta[ms] * (self.T[1] - ts))            # [ batch_size, seq_len ]
        alpha_int = self.kernel._alphas[:, ms].sum(0)
        integ = alpha_int * temp_int * mask
        integ = integ.sum()

        loglik = sumlog - baserate - integ
        
        return loglik