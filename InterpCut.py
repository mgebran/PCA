import os
import numpy as np
import pylab as plb
from pylab import *
import time
from scipy import interpolate
from PyAstronomy import pyasl
lambda_polar=np.arange(4400.,5000.,0.05) # polarbase for A stars Hbeta and some metallic

lst=np.loadtxt("filedat",dtype=str)
ii=0
for i in lst:
	spec=np.loadtxt(i)
	wv=spec[:,0]
	f=spec[:,1]

	interpfunc = interpolate.interp1d(wv,f, kind='linear')
	flux1=interpfunc(lambda_polar)
	np.savetxt("Reduced/"+lst[ii]+".reduced.dat",np.array([lambda_polar,flux1]).T)
	ii+=1
