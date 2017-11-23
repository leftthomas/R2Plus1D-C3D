import numpy as np
from torchnet.meter import meter


class AverageValueMeter(meter.Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.sum += value
        self.n += n

    def value(self):
        n = self.n
        if n == 0:
            mean = np.nan
        elif n == 1:
            return self.sum
        else:
            mean = self.sum / n
        return mean

    def reset(self):
        self.sum = 0.0
        self.n = 0
