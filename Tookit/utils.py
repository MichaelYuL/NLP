#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time   : 2023/4/6 00:08
# @Author : Lixinqian Yu
# @E-mail : yulixinqian805@gmail.com
# @File   : utils.py
# @Project: NLP
import numpy as np
import torch


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # the random number for numpy's methods is fixed
    torch.manual_seed(seed)  # the random number for CPU is fixed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # the rand number for all GPU are fixed


def LOG(mode, text):
    print(f"[{mode}]--> {text}")


if __name__ == '__main__':
    pass