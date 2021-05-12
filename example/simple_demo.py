# -*- coding: utf-8 -*- 
# @Time : 2021/5/12 21:04 
# @Author : lepold
# @File : simple_demo.py


from thetanet import parameter
from thetanet import node_network
from scripts import draw

pm = parameter.Parameter()
pm.all_to_all_interconnected_populations(20, 20, kappa_intra=3., kappa_inter=0.6)
y = node_network.integrate(pm, init=None, console_output=True)
draw.draw_sample_trajectory(y)

