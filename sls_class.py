# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:13:01 2019

@author: Gordon
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

class sls:
    def __init__(self, filename):
        self.get_data(filename)
        
    def get_data(self, filename):
        with open(filename) as csvfile:
            r = csv.reader(csvfile, delimiter = ' ')
            data = np.array([[float(row[0]), float(row[1])] for row in r])
            
        x = data[:,0]      #data
        y = data[:,1]
        v = np.std(y)**2   #variance of y for caculatin penalty
        n = len(x)
        err_arr = np.zeros((n,n))
        a_arr = np.zeros((n,n))
        b_arr = np.zeros((n,n))
        
        #for j  end indices of data, 0 to n-1
        for j in range(n):
            #for i starting indices of this data, 0 to j
            for i in range(j+1):
                #for this segment, i to j, get coefs and error of fit
                a, b, e2 = self.lscoef(x[i:j+1],y[i:j+1])
                
                #store error and coefs for segment i to j in array at [i,j]
                err_arr[i,j] = e2
                a_arr[i,j] = a
                b_arr[i,j] = b
        
        self.x = x
        self.y = y
        self.err_arr = err_arr
        self.a_arr = a_arr 
        self.b_arr = b_arr
        self.v = v

    def lscoef(self, x, y):
        n=len(x)
        if (n == 1):
            return (0.0, y[0], 0.0)
        
        sx = np.sum(x)
        sy = np.sum(y)
        sx2 = np.sum(x ** 2)
        sxy = np.sum(x * y)
        a = (n*sxy-sx*sy)/(n*sx2-sx*sx)    
        b = (sy-a*sx)/n
        e2 = np.sum((y-a*x-b)**2)

        return (a, b, e2)
        
    def find_opt(self, penalty_factor):
        penalty = self.v * penalty_factor
        n = self.err_arr.shape[0]
        
        #for each end index, get minimum error over possible start indices
        opt_arr = np.zeros(n)
        for j in range(n):
            #for this end index, use min error for previous end index and
            #current error to get errors for all possible start indices
            tmp_opt = np.zeros(j+1)
            tmp_opt[0] = self.err_arr[0,j] + penalty
            for i in range(1,j+1):
                tmp_opt[i] = opt_arr[i-1] + self.err_arr[i,j] + penalty
            opt_arr[j] = np.amin(tmp_opt)
        
        #backtrack from opt_arr[n-1] to get segment and coefficients
        opt_coefs = []
        j = n-1
        while j >= 0:
            tmp_opt = np.zeros(j+1)
            tmp_opt[0] = self.err_arr[0,j] + penalty
            for i in range(1,j+1):
                tmp_opt[i] = opt_arr[i-1] + self.err_arr[i,j] + penalty
            i_opt = np.argmin(tmp_opt)
            a_opt = self.a_arr[i_opt,j]
            b_opt = self.b_arr[i_opt,j]
            
            #set boundaries of interval for these coefs in terms of x
            if i_opt <= 0:
                xmin = np.NINF
            else:
                xmin = (self.x[i_opt-1] + self.x[i_opt])/2
            if j >= n-1:
                xmax = np.Inf
            else:
                xmax = (self.x[j] + self.x[j+1])/2     
            opt_coefs.insert(0, (xmin,xmax,a_opt,b_opt))
            j = i_opt-1
            
        self.opt_coefs = opt_coefs
        
    def get_num_segments(self):
        return len(self.opt_coefs)
        
    def get_fit(self,x):
        #get fit using coefs
        n = len(x)
        yfit = np.zeros(n)
        for opt_coef in self.opt_coefs:
            ind = [i for i,elem in enumerate(x) \
                   if elem >= opt_coef[0] and elem <= opt_coef[1]]
            yfit[ind] = x[ind] * opt_coef[2] + opt_coef[3]

        return yfit
        
    def plot_fit(self,xplt):
        yfit = self.get_fit(xplt)
        plt.plot(self.x, self.y, '.')
        plt.plot(xplt, yfit)
        plt.show()