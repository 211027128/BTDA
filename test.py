# -*- coding == utf-8 -*-
# @Time : 2023/3/1 20:45
# @Author : 211027128
# @File : test.py
# @Software :PyCharm
import numpy as np
import torch
a=[4,3,2,1]
c=[0.4,0.3,0.2,0.1]
c=c[-1]/torch.tensor(c)
b=a[-1]/np.array(a)
d=1 - (500 / 500) * (1 - c)
print(d)
print(1)
print(2)
print(3)
print(4)

