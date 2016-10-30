# -*- coding: utf-8 -*-

import math
import random
import matplotlib.pyplot as pyplot
import segleastsquares as sls

#generate data
def gendata():
    n=50
    per = 20
    coef2 = 0.025
    coef4 = 0.0000001
    xarray = range(50)
    yarray = []
    for x in xarray:
        noise = 5*random.random()
        yarray = yarray + [noise+(1+coef2*x*x-coef4*x*x*x*x)*math.sin(2*2.31416*x/per)]
    return (n,xarray,yarray)

#read file of x y pairs, one per line
def readdata(fname):
    xarray = []
    yarray = []

    with open(fname) as f:
        for line in f:
            x,y = line.split()
            xarray = xarray + [float(x)]
            yarray = yarray + [float(y)]
    return (len(xarray), xarray, yarray)
    

inputFlag = True           #true to generate data, false to read from file
filename = 'testdata.txt'    #input text file - two numbers per line:  x y

if (inputFlag):
    n,xarray,yarray = gendata()
else:    
    n,xarray,yarray = readdata(filename)     #get data
    
Cfactor = 0.35       #factor in computing penalty for adding segments

#segmented least squares
yfit = sls.segls(n,xarray,yarray,Cfactor)

pyplot.plot(xarray,yarray,xarray,yfit)
pyplot.show()