#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time   : 2023/3/22 14:47
# @Author : Lixinqian Yu
# @E-mail : yulixinqian805@gmail.com
# @File   : pre_trained_networks.py
# @Project: NLP
from torchvision import models


if __name__ == '__main__':
    alexnet = models.AlexNet()
    print(f"[info] : AlexNet is built!")
