# -*- coding: utf-8 -*- 
# @Time : 2021/5/12 20:52 
# @Author : lepold
# @File : draw.py

import numpy as np
import matplotlib.pylab as plt

def draw_sample_trajectory(y, time=100):
    y = y % (np.pi)
    iteration, N = y.shape[0], y.shape[1]
    sample = np.random.choice(N, size=5)
    fig = plt.figure(figsize=(5, 3), dpi=300)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    data = y[-time:, sample]
    axes.plot(data)
    plt.show()
