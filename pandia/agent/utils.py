import numpy as np


def index_of(value, array):
    return array.index(min(array, key=lambda x:abs(x - value)))

def sample(val):
    if type(val) is list:
        return np.random.uniform(val[0], val[1])
    else:
        return val

def deep_update(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1