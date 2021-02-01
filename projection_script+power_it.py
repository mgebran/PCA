#execute with sudo
from numpy import *
import numpy as np
import joblib
import gc
import os
import math
import pickle
from scipy import interpolate
import time
t=time.time()

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

# Creating the database for Teff logg Mean meta and vsini
sizef=zeros(len(lst), 'Float32')
for i in arange(len(lst)-1):
	tam=len(joblib.load(lst[i]))
	sizef[i+1]=sizef[i]+tam
	print lst[i],"has a size of",tam


teff=[]
logg=[]
vsini=[]
meta=[]
count=0
data=1
mn=0

def del_mem():
	reload(joblib)
	gc.collect()
	os.system("echo 1 > /proc/sys/vm/drop_caches ")
	os.system("echo 2 > /proc/sys/vm/drop_caches ")
	os.system("echo 3 > /proc/sys/vm/drop_caches ")
	os.system("sync ")

for i in lst:
	del_mem()
	print "The dumpfile loading is:",i
	tl=time.time()
	data=joblib.load(i)
	tl=time.time()-tl
	print tl
	nmod=shape(data)[0]
	for i in arange(nmod):
		if count==0:
			mn=data[0][0]
		else:
			mn=(mn*count+data[i][0])/(count+1)
		teff.append(data[i][1])
		logg.append(data[i][2])
		vsini.append(data[i][3])
		meta.append(data[i][4])
		count+=1
	print nmod,count
	del data
	del_mem()
	
savetxt("teff",teff,fmt="%g")
savetxt("logg",logg,fmt="%g")
savetxt("vsini",vsini,fmt="%g")
savetxt("meta",meta,fmt="%g")
savetxt("mean",mn)
t=time.time()-t
print t
print "parameters saved"


i_list=sizef

####################################################################################################################################3
# Calculating the Eingevectors
print "Calculating eigen vectors"
n_wv=size(mn)
n_mod=size(teff)
n_iter=40
data=0
e=[]
def getspec(i):
	global data
	if i in i_list:
		n=where(i_list==i)[0]
		del_mem()
		del(data)
		data=joblib.load(lst[n])
		print lst[n]
		return data[0][0]
	else:
		n=where(i>i_list)[0].max()
		return data[int(i-i_list[n])][0]



def d(r):
	d=zeros(n_mod)
	for i in arange(n_mod):
		d[i]=dot(r,getspec(i)-mn)
	return d

def gen_r(): #function to generate r
	r=np.zeros(n_wv)
	for i in np.arange(n_wv):
		r[i]=random.uniform(-1,1)
	r=r/np.linalg.linalg.norm(r)
	return r


print "Eigen vector 1"

#r1 is the first eigen vector
r1=gen_r()
t=time.time()
t_all=time.time()
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		s+=dot(p,r1)*p
	r1=s/linalg.linalg.norm(s)


t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e1",r1)

print "Eigen vector 2"

#r2 is the second eigen vector
r2=gen_r()
t=time.time()
d1=d(r1)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		s+=dot(p,r2)*p
	r2=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e2",r2)

print "Eigen vector 3"

#r3 is the second eigen vector
r3=gen_r()
t=time.time()
d2=d(r2)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		s+=dot(p,r3)*p
	r3=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e3",r3)


print "Eigen vector 4"

#r4 is the second eigen vector
r4=gen_r()
t=time.time()
d3=d(r3)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		s+=dot(p,r4)*p
	r4=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e4",r4)


print "Eigen vector 5"

#r5 is the second eigen vector
r5=gen_r()
t=time.time()
d4=d(r4)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		s+=dot(p,r5)*p
	r5=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e5",r5)


print "Eigen vector 6"

#r6 is the second eigen vector
r6=gen_r()
t=time.time()
d5=d(r5)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		s+=dot(p,r6)*p
	r6=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e6",r6)


print "Eigen vector 7"

#r7 is the second eigen vector
r7=gen_r()
t=time.time()
d6=d(r6)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		p=p-d6[ii]*r6
		s+=dot(p,r7)*p
	r7=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e7",r7)


print "Eigen vector 8"

#r8 is the second eigen vector
r8=gen_r()
t=time.time()
d7=d(r7)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		p=p-d6[ii]*r6
		p=p-d7[ii]*r7
		s+=dot(p,r8)*p
	r8=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e8",r8)


print "Eigen vector 9"

#r9 is the second eigen vector
r9=gen_r()
t=time.time()
d8=d(r8)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		p=p-d6[ii]*r6
		p=p-d7[ii]*r7
		p=p-d8[ii]*r8
		s+=dot(p,r9)*p
	r9=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e9",r9)


print "Eigen vector 10"

#r10 is the second eigen vector
r10=gen_r()
t=time.time()
d9=d(r9)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		p=p-d6[ii]*r6
		p=p-d7[ii]*r7
		p=p-d8[ii]*r8
		p=p-d9[ii]*r9
		s+=dot(p,r10)*p
	r10=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e10",r10)


print "Eigen vector 11"

#r11 is the second eigen vector
r11=gen_r()
t=time.time()
d10=d(r10)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		p=p-d6[ii]*r6
		p=p-d7[ii]*r7
		p=p-d8[ii]*r8
		p=p-d9[ii]*r9
		p=p-d10[ii]*r10
		s+=dot(p,r11)*p
	r11=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n"
savetxt("e11",r11)


print "Eigen vector 12"

#r12 is the second eigen vector
r12=gen_r()
t=time.time()
d11=d(r11)
for i in arange(n_iter):
	print "Iteration #: %i" %(i+1)
	s=zeros(n_wv)
	for ii in arange(n_mod):
		p=getspec(ii)-mn
		p=p-d1[ii]*r1
		p=p-d2[ii]*r2
		p=p-d3[ii]*r3
		p=p-d4[ii]*r4
		p=p-d5[ii]*r5
		p=p-d6[ii]*r6
		p=p-d7[ii]*r7
		p=p-d8[ii]*r8
		p=p-d9[ii]*r9
		p=p-d10[ii]*r10
		p=p-d11[ii]*r11
		s+=dot(p,r12)*p
	r12=s/linalg.linalg.norm(s)

t=time.time()-t
m,s=divmod(t,60)
h,m=divmod(m,60)
print "Calculation took:"
print "%dh %02dmin %02dsec" % (h, m, s) 
print "\n\n\n"
savetxt("e12",r12)


print "The total time taken is:"
t_all=t_all-time.time()
m,s=divmod(t_all,60)
h,m=divmod(m,60)
print "%dh %02dmin %02dsec" % (h, m, s) 
e=vstack((r1.T,r2.T,r3.T,r4.T,r5.T,r6.T,r7.T,r8.T,r9.T,r10.T,r11.T,r12.T))
e=e.T
savetxt("eigen",e)
print "Eigen vectors saved, projecting coefficients"
#########################################################################################################################################

#CALCULATING THE COEFFICIENTS


teff=loadtxt("teff")
e=loadtxt("eigen")

nmod=size(teff)
mn=loadtxt("mean")
nk=12
p=zeros((nmod,nk),'Float32')

for i in arange(nmod):
	spec=getspec(i)
	for k in arange(nk):
		p[i,k]=dot((spec-mn),e[:,k])

savetxt("coeff",p)
