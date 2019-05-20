# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:22:58 2019

@author: Gordon
"""
import numpy as np
from sls_class import sls

s = sls('testdata.txt')
s.find_opt(penalty_factor=0.35)

n = len(s.x)
minx = np.amin(s.x)
maxx = np.amax(s.x)
dx = 0.33*(maxx-minx)/(n-1)
xplt = np.arange(minx,maxx+dx,dx)

s.plot_fit(xplt)