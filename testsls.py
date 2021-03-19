# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:22:58 2019

@author: Gordon
"""
import numpy as np
from sls_class import sls

# control how segments are allocated to parts of the data,
# either by a given penalty (faster), or by a maximum number of segments

# object creation, read data
s = sls('testdata.txt')

# find the least squares segment coefficients
# if no parameters input, defaults to desired_penalty of 0.35
# to limit the number of segments, set max_num_seg (desired_penalty ignored)
# to set a desired_penalty, do not input max_num_seg to set it to None
penalty, num_segments = s.find_segments()
print("penalty = {0}, num segments = {1}".format(penalty, num_segments))

# fit to a regular grid for plotting
n = len(s.x)
minx = np.amin(s.x)
maxx = np.amax(s.x)
dx = 0.33*(maxx-minx)/(n-1)
xplt = np.arange(minx,maxx+dx,dx)

# plot the segments
s.plot_fit(xplt)
