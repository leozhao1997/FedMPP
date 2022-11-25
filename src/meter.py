"""
    Training Loss/Acc Recording
    Reference to CMU Course 18-661: Intro to ML for Engineer
"""
import numpy as np 

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