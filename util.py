"""
    Training Loss/Acc Recording
    Reference to CMU Course 18-661: Intro to ML for Engineer
"""
import numpy as np 
import torch

class RecordingMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0
        self.list = []
    
    def update(self, increment, count):
        self.qty += increment
        self.cnt += count
        self.list.append(increment/count)
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else: 
            return self.qty/self.cnt

    def get_sum(self):
        return self.qty

    def get_var(self):
        return np.var(self.list)

    def get_top(self, percent):
        temp = np.array(self.list).reshape(-1)
        temp = np.sort(temp)[-int(len(temp)*percent):]
        return np.mean(temp, keepdims=False)

    def get_low(self,percent):
        temp = np.array(self.list).reshape(-1)
        temp = np.sort(temp)[:int(len(temp)*percent)]
        return np.mean(temp, keepdims=False)

    def get_all(self):
        return np.array(self.list).reshape(-1)


class NonNegativeClipper(object):
    """
    References:
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933
    https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/3
    """

    def __init__(self):
        pass

    def __call__(self, module):
        """enforce non-negative constraints"""
        if hasattr(module, '_mu'):
            _mu = module._mu.data
            module._mu.data = torch.clamp(_mu, min=0.)
        # Multivariate Hawkes
        if hasattr(module, '_alphas'):
            _alphas = module._alphas.data
            module._alphas.data = torch.clamp(_alphas, min=1e-5)
        if hasattr(module, '_beta'):
            _beta  = module._beta.data
            module._beta.data  = torch.clamp(_beta, min=1e-5)