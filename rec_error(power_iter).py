import numpy as np
from pylab import *
import joblib
import gc
import os
import math
import pickle
from scipy import interpolate
import time


#e=eigen vectors
#p=projected coefficients
#mn=spectra mean


lst=["DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-2.0-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.9-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.8-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.7-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.6-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.5-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.4-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.3-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.2-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.1-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-1.0-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.9-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.8-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.7-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.6-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.5-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.4-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.3-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.2-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.1-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300-0.0-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.1-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.2-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.3-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.4-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.5-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.6-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.7-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.8-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+0.9-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.0-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.1-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.2-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.3-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.4-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.5-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.6-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.7-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.8-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+1.9-Vmicr2-Resolution76000",
"DumpfilePolarBase-4450-4990-6800-11000-100-2-5-0.1-vsini-0-300+2.0-Vmicr2-Resolution76000",]

i_list=[]
i_list.append(0)
kk=1
while (kk<41):
	i_list.append(kk*54653)
	kk+=1

data=0
e=np.loadtxt("eigen")
p=np.loadtxt("coeff")
mn=np.loadtxt("mean")
def getspec(i):
	global data
	if i in i_list:
		n=np.where(i_list==i)[0]
		del_mem()
		del(data)
		data=joblib.load(lst[n])
		print lst[n]
		return data[0][0]
	else:
		n=np.where(i>i_list)[0].max()
		return data[i-i_list[n]][0]


def del_mem():
	reload(joblib)
	gc.collect()
	os.system("echo 1 > /proc/sys/vm/drop_caches ")
	os.system("echo 2 > /proc/sys/vm/drop_caches ")
	os.system("echo 3 > /proc/sys/vm/drop_caches ")
	os.system("sync ")


rec_er=np.zeros(np.size(p,axis=0))
for i in np.arange(np.size(rec_er)):
	spec_rec=np.dot(e,p[i])+mn
	spec=getspec(i)
	err=np.abs(spec_rec-spec)/spec
	rec_er[i]=err.mean()	
rec_er_all=rec_er.mean()


