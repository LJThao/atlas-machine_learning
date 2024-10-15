#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    confusion = np.load('/root/atlas-machine_learning/supervised_learning/classification/data/confusion.npy')['confusion']

    np.set_printoptions(suppress=True)
    print(sensitivity(confusion))