# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:22:58 2019

@author: Gordon
"""
import numpy as np
from sls_class import sls

s = sls('testdata.txt')

# control how segments are allocated to parts of the data,
# either by a given penalty, or by a maximum number of segments

# if want a certain penalty (and it will be faster), set this
# otherwise set to None to use a max number of segments
desired_penalty = None  #0.35  

# if desired_penalty is None, set these:
max_num_seg = 3
penalty_start = 0.1
penalty_inc = 0.05

if desired_penalty:
    penalty = desired_penalty
    s.find_opt(penalty_factor=penalty)
else:
    penalty = penalty_start-penalty_inc
    num_seg = np.inf
    while num_seg > max_num_seg:
        penalty += penalty_inc
        s.find_opt(penalty_factor=penalty)
        num_seg = s.get_num_segments()

print("penalty = {0}, num segments = {1}".format(penalty, s.get_num_segments()))

# fit to a regular grid for plotting
n = len(s.x)
minx = np.amin(s.x)
maxx = np.amax(s.x)
dx = 0.33*(maxx-minx)/(n-1)
xplt = np.arange(minx,maxx+dx,dx)

s.plot_fit(xplt)