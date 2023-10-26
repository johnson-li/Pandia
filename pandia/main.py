import os
import re


def test():
    a = {'1': '1', '2': '2'}
    b = {'b': a}
    b.update({'b': {'1': '1'}})
    print(b)
    

if __name__ == '__main__':
    test()
