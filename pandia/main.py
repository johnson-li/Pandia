import os
import re


def test():
    try:
        print('asdf')
        return 1
    finally:
        print(123)
        return 2
    

if __name__ == '__main__':
    a = test()
    print(a)
