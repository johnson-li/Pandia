import numpy as np


def index_of(value, array):
    return array.index(min(array, key=lambda x:abs(x - value)))

def sample(val):
    if type(val) is list:
        return int(np.random.uniform(val[0], val[1]))
    else:
        return val