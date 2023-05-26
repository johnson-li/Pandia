def index_of(value, array):
    return array.index(min(array, key=lambda x:abs(x - value)))