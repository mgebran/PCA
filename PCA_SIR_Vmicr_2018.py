#PCA functions
"This is a module that is used to for PCA on synthetic spectra. It is quite easy to use and implement. _path is the path to the spectra database, where each database should be put in a folder called by its name. The databases should/might include txt files of: eigen values, coefficients, the spectra's mean value, teff logg and vsini of each spectrum, and a file called mat that contains all the spectra, each on a line, in the corresponding database. setup function should be executed when any database should be used. generate_database creates databases from dumbfiles. Enjoy!"

__version__=1.3

import numpy as np
from scipy import interpolate
import random as rd
import linecache
import functools
import os
from pylab import *
import subprocess
import sys
from numpy import linalg as LA
from astropy.stats import sigma_clip as sgc 
from PyAstronomy import pyasl


_path="./Database/"  #Path to spectra database
_Name_dp=None  #Name of the dumbfile
_gl=False

_synspec_path="/home/mgebran/Desktop/synspec48_3/"
_synspec="synspecInver-SIR_Vmicr.py"

lstdir=os.listdir(_path)
database_list=[] #list of all the dumbfile in the database.
for i in lstdir:
	if "eigen_"+i in os.listdir(_path+i):
		database_list.append(i)

del(lstdir)

ldelta=np.arange(4030., 4194.95, 0.05) #ldelta database wavelength
ldelta_red=np.arange(4050.,4149.95,0.05) #reduced ldelta (DumpfileHdelta-4050-4150-5000-15000-2-5)
lmet=np.arange(4336., 4582, 0.05) #met database wavelength
lmet_red=np.arange(4400,4550,0.05) #reduced met database wavelength
lS4N=np.arange(5000.,5400.,0.05) #S4N database wavelength
lS4N_red=np.arange(5130,5280,0.05) #reduced S4N wavelength
ldm=np.arange(5780., 6000., 0.05)  # dM
lambda_polar=np.arange(4450.,4990,0.05) # polarbase for A stars Hbeta and some metallic
lambda_polar_hdelta=np.arange(4780.,4970,0.05) # Polarbase for A stars Hbeta 
lambda_polar_met=np.arange(4450.,4750,0.05)# Polarbase for A stars metallic
lambda_WEAVE_blue=arange(4450.,4600.,0.05)#WEAVE for blue arm metallic
lambda_WEAVE_red=arange(6200.,6750.,0.05) # WEAVE red arm for Halfa
lambda_WEAVE_BP=arange(4200.,4600.,0.05)#WEAVE for blue arm metallic
lambda_WEAVE_LR=arange(4000.,5000.,0.4)#WEAVE Low resultion R=5750
lambda_WEAVE_LR_red=arange(5800.,7500.,0.2)#WEAVE Low resultion R=5750
lambda_WEAVE_HR_red=arange(6000.,6350.,0.05)#WEAVE Low resultion R=21000
lambda_K1=arange(5000.,5400.,0.10)#31500
lambda_K2=arange(5000.,6000.,0.10)#31500
lambda_K3=arange(5800.,6000.,0.10)#31500
#for 2regions
lambda1=arange(4450.,4550.,0.05) 
lambda2=arange(4750.,4980.,0.05) 
lambda3=np.concatenate((lambda1,lambda2),axis=0)
lambdaK=arange(5070., 5340, 0.05)
#for G stars
lambdaG1=np.arange(5760.,5786.,0.04)
lambdaG2=np.arange(5799.,5881.,0.04)
lambdaG3=np.arange(5900.,5976.,0.04)
lambdaG4=np.arange(5997.,6073.,0.04)
lambdaG5=np.arange(6103.,6185.,0.04)
lambdaG6=np.arange(6327.,6405.,0.04)
lambdaG7=np.arange(6431.,6523.,0.04)
lambdaG8=np.arange(6550.,6643.,0.04)
lambdaG9=np.arange(6676.,6767.,0.04)
lambdaG10=np.arange(7075.,7175.,0.04)
lambdaG11=np.arange(7220.,7320.,0.04)
lambdaG12=np.arange(7375.,7471.,0.04)
lambdaG13=np.arange(7689.,7798.,0.04)
lambdaG14=np.arange(7863.,7968.,0.04)
lambdaG15=np.arange(8040.,8139.,0.04)
lambdaG16=np.arange(5760.,8139.,0.04)


lambdaAPO=arange(4500., 6800., 0.15)  # APO


def path():
	return 	_path
def list_all():
	"""Function that lists all the database names. It reads the dabase_list array and prints it. The number of each database could be passed to the setup function instead of the name of the database as a string"""
	for i in np.arange(np.size(database_list)):
		print str(i)+" - "+database_list[i]

def setup(Name_dp,wv=None,_res=None,h="synspec"):
    """A required function that takes name of dumpfile, desired resolution of synthetic spectra, and wavelength grid as optional input (in case interpolation is necessary)."""
    global _Name_dp,e,p,mean,meta,teff,logg,vsini,micro,nk,dbnspec,wavelength,_meta,res,_h
    if h is "synspec" or h is "mat" or h is "rec":
	_h=h
    else:
	raise RuntimeError("'"+h+"' is a wrong input. Please replace by either 'mat' or 'rec', or leave empty.")
    if _res is None:
	raise RuntimeError('Please input resolution')
    else:
	res=_res
    if isinstance(Name_dp,int) is True:
	_Name_dp=database_list[Name_dp]
    else:
	_Name_dp = Name_dp
    _meta=False
    wavelength=wv
    e=np.loadtxt(_path+_Name_dp+"/eigen_"+_Name_dp)
    p=np.loadtxt(_path+_Name_dp+"/coeff_"+_Name_dp)
    mean=np.loadtxt(_path+_Name_dp+"/mean_"+_Name_dp)
    teff=np.loadtxt(_path+_Name_dp+"/teff_"+_Name_dp)
    logg=np.loadtxt(_path+_Name_dp+"/logg_"+_Name_dp)
    vsini=np.loadtxt(_path+_Name_dp+"/vsini_"+_Name_dp)
    micro=np.loadtxt(_path+_Name_dp+"/micro_"+_Name_dp)
    if "meta_"+_Name_dp in os.listdir(_path+_Name_dp):
	meta=np.loadtxt(_path+_Name_dp+"/meta_"+_Name_dp)
	_meta=True
    nk=np.size(e,axis=1) # Number of eigenvectors
    dbnspec=np.size(p,axis=0)  # Numer of spectra in learning database
    test=np.size(e,axis=0) # Wavelength test
    if (test==np.size(wavelength)) is False and wavelength!=None:
	raise RuntimeError('Check wavelength grid')

def setup_required(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
	global _Name_dp
	if _Name_dp is None:
	    raise RuntimeError('Setup Required')
	return func(*args, **kwargs)
    return wrapped

def stat():
    """Function that gives the status of PCA module. It should print the Database currently being used, and if linecache is being used in memory"""
    if _Name_dp is None:
	print "No database is specified. Please load database using 'setup' function"
    else:
	print "\n\n"
	if _gl is False:
	    print "The database currently in use is: "+_Name_dp+", linecache is not loaded on memory"
	    print "\n"
	else:
	    print "The database currently in use is: "+_Name_dp+", linecache is loaded on memory"
	    print "\n"
	if wavelength is not None:
	    print "The wavelength range is:"
	    print wavelength
	    print "\n"	
	print "The teff range is:"
	print np.unique(teff)
	print "\n"
	print "The logg range is:"
	print np.unique(logg)
	print "\n"
	print "The Vsini range is:"
	print np.unique(vsini)
	print "\n"
	print "The Microturbulence range is:"
	print np.unique(micro)
	print "\n"

	if _meta:
	    print "The metalicity range is:"
	    print np.unique(meta)
	    print "\n"
	


def generate_database(Name_dp,spec_mat=True):
	"Function that generates the database (read below the np.savetxt files), from dumbfiles. The dumbfiles should be in _path, in folders called by their names. If spec_mat=False, the script won't save the matrix file of the database ('mat_'+Name_dp), default is True. RELOAD MODULE ONCE FINISHED"""
	if Name_dp in database_list:
		raise RuntimeError("The Database is already generated!")
	import pickle
	import time

	t=time.time()

	######################
	#  READING THE DATA  #
	######################

	path=_path+Name_dp+"/" # Enter path to dumpfile


	Fic=open(path+Name_dp,'r')
	grid=pickle.load(Fic)
	Fic.close()
	print "\n\n\n\n\n"
	print "==========================================================================="
	print "Synthetic data read from pickle"
	print "===========================================================================\n"

	nmod=len(grid) # Number of spectra in learning database
	nwav=len(grid[0][0]) # Numer of data point in each spectra

	mat=np.zeros((len(grid), nwav), 'Float32') # nmod by nmwav matrix of learning database matrix
	teff=[] # Teff of each spectrum
	logg=[] # logg of each spectrum
	vrot=[] # vrot of each spectrum
	micro=[]
	for i in np.arange(nmod):
	    mat[i,:]=grid[i][0]
	    teff.append(grid[i][1])
	    logg.append(grid[i][2])
	    vrot.append(grid[i][3])
	    micro.append(grid[i][5])
	print "==========================================================================="
	print  "Matrix of data, vectors of Teff and logg done, proceeding to SVD"
	print "===========================================================================\n"

	###############################################################
	# Performing SVD. C.C_trans is the variance-covariance matrix #
	############################################################### 

	mn=np.mean(mat, axis=0)
	C=mat-mn
	e, s, aaa=np.linalg.linalg.svd(np.dot(np.transpose(C),C), full_matrices=False)

	print "==========================================================================="
	print "SVD done"
	print "===========================================================================\n"

	####################################
	# Creating Projection Coefficients #
	####################################
	nk=12
	p=np.zeros((nmod,nk),'Float32')
	for k in np.arange(nk):
	    for i in np.arange(nmod):
		p[i,k]=np.dot((mat[i,:]-mn),e[:,k])

	#################################
	# Saving data to specified path #
	#################################

	print "==========================================================================="
	print "Projected coefficients created"
	print "===========================================================================\n"
	print "==========================================================================="
	print "Final step: saving files"
	print "===========================================================================\n"

	np.savetxt(path+"coeff_"+Name_dp,p)
	np.savetxt(path+"eigen_"+Name_dp,e[:,:12])
	np.savetxt(path+"teff_"+Name_dp,teff)
	np.savetxt(path+"logg_"+Name_dp,logg)
	np.savetxt(path+"mean_"+Name_dp,mn)
	np.savetxt(path+"vsini_"+Name_dp,vrot)
	np.savetxt(path+"micro_"+Name_dp,micro)
	if spec_mat:
		np.savetxt(path+"mat_"+Name_dp,mat)

	t=(time.time()-t)
	m, s = divmod(t, 60)
	h, m = divmod(m, 60)
	print "Calculation took:"
	print "%dh %02dmin %02dsec" % (h, m, s)
	print "\n"
	print "____________RELOAD PCA MODULE____________"

def noise(f,sn):
    """Function to create Gaussian noise. f should be a numpy array, sn is the desired Signal to Noise ratio"""
    if isinstance(f,np.ndarray) is False:
	raise RuntimeError('Array should be a numpy array')
    z=f.copy()
    for i in np.arange(np.size(z)):
	r=100
	while (r>1.0 or r<0.0):
	    x=2*rd.uniform(0,1)-1
	    y=2*rd.uniform(0,1)-1
	    r=x**2+y**2
	gaussnoise=x*np.sqrt(-2*np.log(r)/r)
	z[i]=z[i]+(1./sn)*gaussnoise
	if z[i]<0.0:
	    z[i]=0.0
    return z

@setup_required
def index(t,g,v,mi,m=None):
    """Function that returns the index of the spectrum with teff=t, logg=g, vsini=v, micro=mi"""
    a=np.where(teff==t)[0]
    b=np.where((logg>g-0.001)&(logg<g+0.001))[0]
    c=np.where(vsini==v)[0]
    e=np.where(micro==mi)[0]
    if _meta:
	d=np.where((meta>m-0.001)&(meta<m+0.001))[0]
	ab=np.intersect1d(a,b)
	abc=np.intersect1d(ab,c)
	abcd=np.intersect1d(abc,d)
	abcee=np.intersect1d(abcd,e)
	return abcde[0]
    else:
	ab=np.intersect1d(a,b)
	abc=np.intersect1d(ab,c)
	abce=np.intersect1d(abc,e)
	return abce[0]

@setup_required
def getparam(ind):
    """Function that returns the parameters (teff, logg, vsini) of the specified ind"""
    if isinstance(ind,np.ndarray) is True:
	all_param=np.zeros((np.size(ind),4))
	if _meta:
	    all_param=np.zeros((np.size(ind),5))
	for i in np.arange(np.size(ind)):
	    all_param[i]=getparam(ind[i])
	return all_param
    else:
	if _meta:
	    a=np.zeros(5)
	    a[0]=teff[ind]
	    a[1]=logg[ind]
	    a[2]=vsini[ind]
	    a[3]=meta[ind]
	    a[4]=micro[ind]
	    return a
	else:
	    a=np.zeros(4)
	    a[0]=teff[ind]
	    a[1]=logg[ind]
	    a[2]=vsini[ind]
	    a[3]=micro[ind]
	    return a

@setup_required
def getcoeff(spec):
	"""Function that computes and return projected coefficients of the spectrum 'spec'. In case a single integer index is given, function returns coefficients of index from synthetic database."""
	if isinstance(spec,int) is True:
		return p[spec]
	else:
		z=spec.copy()
		rho=np.zeros(nk)
	    	for k in np.arange(nk): # Calculating projected coefficients of observed spectrum
			rho[k]=np.dot((z-mean),e[:,k])
		return rho


@setup_required
def rec_spec(coeff):
	"""Function that returns a reconstructed spectrum from given coefficients (mean is added). In case a spectrum is passed, 'getcoeff' is called to compute projected coefficients, and then a reconstructed spectrum is passed"""
	if np.size(coeff)==nk:
		spec_rec=np.dot(e,coeff)+mean
		return spec_rec
	elif np.size(coeff)==np.size(wavelength):
		cfc=getcoeff(coeff)
		return rec_spec(cfc)
		

@setup_required
def inversion(f,interp=False,wv=None,list_ind=False,cont=False):
    """Function that returns the index of the least difference. If interp=True, wv should be passed as the current wavelength of f. If list_ind is passed as True, a list of indices with increasing difference of coefficient is returned. If cont=True, a continuum correction is applied to the spectrum, according to Gazzano 2010 http://arxiv.org/pdf/1011.5335v1.pdf"""
    z=f.copy()
    global dif,rho
    if isinstance(f,np.ndarray) is False:
	raise RuntimeError('Array should be a numpy array')
    if interp is True:
	if wavelength==None:
	    raise RuntimeError('Specify wavelength in setup')
	interpfunc = interpolate.interp1d(wv,f, kind='linear')
	flux1=interpfunc(wavelength)
    	z=flux1.copy()
    if cont is True:
	return(_continuum_test(z))
    rho=getcoeff(f)
    dif=np.zeros(dbnspec)
    for l in np.arange(dbnspec):
	for k in np.arange(nk):
	    dif[l]=dif[l]+(p[l,k]-rho[k])**2
    if list_ind is True:
	return np.argsort(dif)
    return np.argmin(dif)

@setup_required
def _continuum_test(f,return_spec=True):
    from astropy.stats import sigma_clip
    spec_i=f.copy()
    n=np.arange(np.size(spec_i)) 
    ind_o=inversion(spec_i) #index of first obtained
    count=0
    eps=10**(-4)
    rfitted=np.ones(np.size(spec_i))+1	
    while((eps<np.abs(rfitted.min()-1)) and (eps<np.abs(rfitted.max()-1)) and count<30):
	spec_o=getspec(ind_o,_h) #spectrum of obtained 

	ratio1=spec_i/spec_o #ratio between "re"normalized and obtained spectra
	
	ratio=sigma_clip(ratio1,sigma=2) #clipping ratio from line   
	rfit=np.polyfit(np.where(ratio.mask==False)[0],ratio.data[~ratio.mask],2)

	rfitted=np.polyval(rfit,n) #low order fit for continuum

	spec2=spec_i/rfitted # renormalizing spectrum

	ind_o=inversion(spec2) #index obtained
	spec_i=spec2
	count+=1
    if return_spec:    
	return(ind_o,spec2)
    else:
	return(ind_o)

@setup_required
def getspec(ind,how='synspec'):
    """Function that returns the flux of a spectrum of index=ind. The 'how' variable is how the spectrum could be retrieved. If how='synspec', then PCA calls synspec to compute the flux. If how='mat' then the module gets the desired spectrum from the mat file if it is available. If how='rec' then the spectrum is reconstructed from the available PCA coefficients, back into the wavelength space."""
    if ind>(dbnspec-1) or ind<0:
	raise RuntimeError('Index is not in database')
    global _gl
    if how=='synspec':
	pp=getparam(ind)
	step_w=round((round(wavelength.max(),2)-round(wavelength.min(),2))/(len(wavelength)),2)
	if _meta:
		subprocess.call([sys.executable,_synspec_path+_synspec,str(res),str(wavelength.min()),str(wavelength.max()),str(pp[0]),str(pp[1]),str(pp[2]),str(pp[3]),str(pp[4]),str(step_w)]) #res,min_wv, max_wv, teff, logg, vsini, meta, micro
	else:
		subprocess.call([sys.executable,_synspec_path+_synspec,str(res),str(wavelength.min()),str(wavelength.max()),str(pp[0]),str(pp[1]),str(pp[2]),"0",str(pp[3]),str(pp[4]),str(step_w)]) #res,min_wv, max_wv, teff, logg, vsini, micro
	spec=np.loadtxt(_synspec_path+"PCAspec")
	return spec
    elif how=='rec':
	return rec_spec(p[ind])
    if _gl==False:
	print "Loading spectrum, please be patient..."
    f=linecache.getline(_path+_Name_dp+"/mat_"+_Name_dp,ind+1)
    f=np.fromstring(f,sep=" ")
    _gl=True
    return f

@setup_required
def rv_cor(flux,wv,rv):
	"""Function that returns corrected RV flux"""
	c=3.0*10**5 #speed of light
	RV_wavelength=wv*(1.-rv/c)
	try:
		f=interpolate.interp1d(RV_wavelength,flux,kind='linear')
		return(f(wavelength))
	except ValueError:
		f=interpolate.interp1d(RV_wavelength,flux,kind='linear',bounds_error=False,fill_value=1)
		return(f(wavelength))


def continuum(n):
    """Function that returns a continuum like polynomial, of n number of points"""
    rang=np.arange(n) 

    n1=rd.uniform(0.95,1.1)
    n2=rd.uniform(0.95,1.1)
    n3=rd.uniform(0.95,1.1)
    n4=rd.uniform(0.95,1.1)
    n5=rd.uniform(0.95,1.1)

    l1=rd.uniform(0,n-1)
    l2=rd.uniform(0,n-1)
    l3=rd.uniform(0,n-1)
    l4=rd.uniform(0,n-1)
    l5=rd.uniform(0,n-1)

    p=np.array([[l1,n1],[l2,n2],[l3,n3],[l4,n4],[l5,n5]]) #points to define continuum
    fit=np.polyfit(p[:,0],p[:,1],2) # Low order "fake" continuum
    pfit=np.polyval(fit,rang)
    return pfit

def im_show(x,y,t=None):
	"""Function that return the matrix of the imshow image. x and y are the arrays to be plotted, t is the the array containing all the possible values of x and y"""
	if t==None:
		t=np.unique(x)
	mat=np.zeros((np.size(t),np.size(t)))
	for i in np.arange(np.size(x)):
		mat[np.where(x[i]==t)[0][0],np.where(y[i]==t)[0][0]]+=1
	return(mat)
####################################################################################
################Here We have the functions for SIR##################################
####################################################################################


#This is a function that is used after we create the reduced data it creates a synthetic spectra grid which resembles a one imported but in the reduced form
#This technique uses less memory hence it is considered more optimized than that np.loadtxt

def grider(spectra_file,name_teff,name_logg,name_vsini,name_met,name_micro):
    Fe_H=[]
    f=open(name_met)
    for line in f:
        val=line.replace('\n',"")
        Fe_H.append(float(val))
    f.close()   


    Teff=[]
    f=open(name_teff)
    for line in f:
        val=line.replace('\n',"")
        Teff.append(float(val))
    f.close()


        
    logg=[]
    f=open(name_logg)
    for line in f:
        val=line.replace('\n',"")
        logg.append(float(val))
        
    f.close()

        
    vsini=[]
    f=open(name_vsini)
    for line in f:
        val=line.replace('\n',"")
        vsini.append(float(val))
    
    f.close()
    
    
    micro_t=[]
    f=open(name_micro)
    for line in f:
        val=line.replace('\n',"")
        micro_t.append(float(val))
	
    f.close()


    red_grid=[]
    j=0
    f=open(spectra_file)
    
        
    for line in f:
        stuff=[]
        val=""    
        for i in line:
            if i==(" "):
                stuff.append(float(val))
                val=""
                continue
            
            val=val+i
            
            if i==("\n"):
                val=val.replace('\n',"")
                stuff.append(float(val))
                    
            
            
    
        spec=np.array(stuff)
        gate=spec,Teff[j],logg[j],vsini[j],Fe_H[j],micro_t[j]
        red_grid.append(gate)
        j=j+1
    
    f.close()
#SIR requires to sort the parameters in increasing so the built-in function "set" removes all the repeated values and we will transform it into a list to use it later on  

    
    
    
    return red_grid   
    
#############################################################################################

#this function takes the m by n matrix sorted by the grid and will give us the empirical matrix in addition to the mean vector of the spectra

#"A" is the sorted matrix made up of only fluxes




def cov_mat(A):
	x_mean=np.mean(A,axis=0) #finds the mean of the spectras where add the spectra and divides them by how many spectras you have
	m=len(A)
	n=len(A[0])
	S=A-x_mean   #S=x-x_mean
		
	siggma=np.matrix(np.zeros((n,n)))	
	
	for i in range(m): #here we will create the empirical covariance matrix sigma=(1/m)sum[ (x-x_mean)T * (x-x_mean)
		siggma=siggma + S[i].T * S[i] 
	
		
	siggma=(1/float(m))*siggma

	return x_mean,siggma

######################################################################################################################


def trenche_mean(A,box):
    n=len(np.array(A[0])[0]) #number of wavelenghts
    
        
    box=sorted(box) 
    H=len(list(set(box))) #number of slices
    means=np.matrix(np.zeros((H,n))) #creates a matrix of zeros
    
    box_count={i:box.count(i) for i in box} # claculates how many times an element is repeated
    
    a=0
    k=0
    for i in sorted(list(set(box))):
        B=A[a:a+box_count[i]]
        a=a+box_count[i]
        
        means[k]=np.mean(B,axis=0)
        k=k+1
        
        num=[]
        for i in box_count.keys():
            num.append(box_count[i])
        
    return means,num,sum(np.array(num))    



#we calculate averages of slices based on the parm thus whatever the numbers of repetions are they will be caluculated appropriately
#we send this to another function 

######################################################################################################################



def with_cov_mat(means,x_mean): #within covarianc matrix
    H=len(means[0]) #number of slices
    n=len(np.array(x_mean[0])[0]) #number of wavelenghts
    gamma=np.matrix(np.zeros((n,n))) #Initialize the gamma matrix or within covariancce matrix

    xs=means[0]
    nh=means[1]
    N=float(means[2])    


    for i in range(H): #we fill the gamma matrix
        S=xs[i] - x_mean

        gamma=gamma + (nh[i]/N)*(S.T*S)

    
    return gamma




######################################################################################################################


#This is the thikhonov-ridge matrix which we will regularize. Hence as an input it requires 3 things: The empirical matrix or sigma,the within slice covariance matrix "gamma",the regularization parameter "delta"
#Here we are using a built in function for finding the inverse of the matrix thick using the linalg package from numpy

def thikhonov(empirical,GAMMA,delta):
    wave=len(empirical) #getting the dimension of the spectrum
    thikh=(empirical*empirical)+(delta*np.matrix(np.identity(wave)))  
    thikh_inv=LA.inv(thikh)
    ridge=thikh_inv*empirical*GAMMA
    
    return ridge




######################################################################################################################



#this function calculates the largest eigenvalue and it's corresponding eigenvector 
#here we are using the power method inhenced with the raylaigh quotient to converge faster
def eig(A):
    m=len(A)
    x0=np.matrix(np.ones(m)).T
    iter=200
    eig0=0

    for k in range(1,iter):
        #finding the eigenvector with largest eigenvalue
        a0=A*x0
        x1=a0/np.max(a0)

        #finding the Rayleigh quotient    
        b1=(A*x1).T*x1
        c1=x1.T*x1
        eig1=b1/c1

        x0=x1
        if ((abs((eig1-eig0)/eig1))<0.0001):break #estimation error of 0.01%
        eig0=eig1
            
    return x1.T,k #returns in in matrix form to be easily used for projection later




######################################################################################################################




#this function is a built in numpy function which does picewise linear interpolation it is very useful in our work becuase the number of elements being converted compared to the synthetic spectra are small and it is optimized for the best way. After testing my own piecewise interpolation technique they seemed equivelent but one issue arrised that the interpolated x-vector should be inputted already in order with it's correcponding y-vector. Hence I used a dictionary to fill,itemize and create and ordered x-vector with it's corresponding y

#Note that in our work the x_h_proj are the projected values of the average of the synthetic spectra of each slice on betta(delta) and the y is the repective average parameter of each slice 

#As a result it returns a float number or array according to the input of the observed,experimental, or to be interpolated data.


def sor_interp(x_obs,x_h_proj,y_h):
    a={x_h_proj[i]:y_h[i] for i in range(len(x_h_proj))}
    x=[]
    y=[] 
    for key in sorted(a.iterkeys()):
        x.append(key)
        y.append(a[key])
        
    

    return np.interp(x_obs,x,y) 





######################################################################################################################


#This function uses the best fitted parameteres to re-normalize the spectra

def gazzano(spec_i,spec_o):
    lamba=len(spec_i)
    n=np.arange(lamba)
    count=0
    eps=1e-4
    rfitted=np.ones(lamba)+1
    while((eps<np.abs(rfitted.min()-1)) and (eps<np.abs(rfitted.max()-1)) and count<15):
        ratio1=spec_i/spec_o
        ratio=sgc(ratio1,sigma=2)
        rfit=np.polyfit(np.where(ratio.mask==False)[0],ratio.data[~ratio.mask],2)

        rfitted=np.polyval(rfit,n) #low order fit for continuum
        spec2=spec_i/rfitted # renormalizing spectrum
        spec_i=spec2
        count+=1
    return spec2        




######################################################################################################################

#PCA is well known for dimension reduction hence we will use it to reduce the dim of the syn spectra


def PCA(grid,nk):
    nmod=len(grid) # Number of spectra in learning database
    nwav=len(grid[0][0]) # Numer of data point in each spectra

#--- common wavelength grid

    mat=np.zeros((len(grid), nwav), 'Float32') # nmod by nmwav matrix of learning database matrix
    #teff=[] # Teff of each spectrum
    #logg=[] # logg of each spectrum
    #vrot=[] # vrot of each spectrum
    #meta=[] # vrot of each spectrum

#Dumps the parameters of each spectrum
    for i in np.arange(nmod):
        mat[i,:]=grid[i][0]   #Fills the matrix with the spectra 
        #teff.append(grid[i][1]) # fills the lists with parameters
        #logg.append(grid[i][2])
        #vrot.append(grid[i][3])
        #meta.append(grid[i][4])    
        
    
    
    mn=np.mean(mat, axis=0) #calculates the mean of the spectra
    C=mat-mn
#Doing SVD
    e, s, aaa=np.linalg.linalg.svd(np.dot(np.transpose(C),C), full_matrices=False)
    


    p=np.zeros((nmod,nk),'Float32')
    for k in np.arange(nk):
        for i in np.arange(nmod):
            p[i,k]=np.dot((mat[i,:]-mn),e[:,k])
            
        
    pc=e[:,0:nk]
    
    return p,pc
    
    


######################################################################################################################

    
    
    
###########################    
######GRSIR FUNCTION####### 
###########################
    
#*******************************************************
#***calculating the regularized tikhonov ridge matrix***
#*******************************************************

#the function GRSIR is a special function used for the purpose of this script to make the computation easier
#This function takes as an input:

# d--> regularization parameter
# xh_bar --> is the average of each slice 
# PARM --> contains the parameter range of synthetic spectra
# select -->is the list that contains some selected spectra from variable "grid"  and attached to t a spectra where we added noise to
# SIG --> empirical covariance matrix
# GAM --> within slice covariance matrix
# parm_i
 




def GRSIR(d,xh_bar,PARM,select,SIG,GAM,GRID,parm_i):
    THICK_RID=thikhonov(SIG,GAM,d) #this creates the matrix that we will find it's eigenvalue note that it is composed of 
					 # ( (sigma^2 +delta*I)^{-1} ) * sigma * gamma   




#We will find the largest eigenvector of the THICK-RID matrix which is a regualarized matrix  based on the power -iteration mode  with Rayleigh quotient which does mazimum of 200 iteration. Not that the test have shown that when it climbs over the 100 iteration the NRMSE increases 

#It is worthy to note that when the regularization parameter is so small to converge to eigenvalues takes time

#we will use the modal empx which has the function nad produces number of iterations and the normalized eigenvector


    betta,iteration=eig(THICK_RID) #This the 
    betta=betta/LA.norm(betta)


#in case you wanted to plot the eigenvectors 
    #betta_array=np.array(betta)[0]
    #la=range(len(betta_array))
    #plt.plot(la,betta_array)
    #plt.xlabel(' # of wavelength')
    #plt.ylabel('coffecient of the basis vector betta')
    #plt.show()



#********************************************
#***Estimating the functional relationship***
#********************************************

#creating the projected values of the slices where  x_proj will be a vector composed of the projected data 
#Not that xh_bar is a matrix which contains the average of each slice so by simply doing matrix multiplication the newly transformed vector will be the projected data that we will use to do interpolation later on.
    x_proj=xh_bar*betta.T

    x_proj=np.array(x_proj.T)[0]
    

    #plt.plot(x_proj,PARM,'*')
    #plt.xlabel("x-proj")
    #plt.ylabel("PARM")
    #plt.show()
    


#Now we will interpolate some observed data we can use our synthetic specra as an initial test
#Or late we can add some noise to it.
#I will select some random values from the grid load them and project them to have
#a set of observed spectra for testing



#************************************************************************
#******generating the test data to choose the regularization vector *****
#************************************************************************

    x_obs=[]
    y_theo=[]  
	#using the select list which contains the indexes of some slected spectra 
	#will help us to choose the best regularization parameter and we will exytract and project the observed data 
	#in the work of wael there was a graph for noise so to choose the best regularizatio parameter we should add that noise and do tests	 	based on that
	#This will be fed by the list with tuples called "lister which contains the test data    

    for i in range(len(select)):
        a=np.matrix(select[i][6])
        a=float(a*betta.T)
        x_obs.append(a)
        
        b=np.matrix(select[i][parm_i])
        y_theo.append(float(b))


    

    y_obs=sor_interp(x_obs,x_proj,PARM)
    y_theo=np.array(y_theo)

    



#Now we should test or results based on the criteria of NRMSE and SIRC 
    
    y_theo_mean=np.mean(y_theo)
    A=(y_obs - y_theo)**2
    B=(y_theo - y_theo_mean)**2

    A=np.sum(A)
    B=np.sum(B)

    
    NRMSE=np.sqrt(A/B)  

    return NRMSE

    
    

######################################################################################################################

def GRSIR_best(d,xh_bar,PARM,select,SIG,GAM,GRID,parm_i):
    THICK_RID=thikhonov(SIG,GAM,d) #this creates the matrix that we will find it's eigenvalue note that it is composed of 
					 # ( (sigma^2 +delta*I)^{-1} ) * sigma * gamma   




#We will find the largest eigenvector of the THICK-RID matrix which is a regualarized matrix  based on the power -iteration mode  with Rayleigh quotient which does mazimum of 200 iteration. Not that the test have shown that when it climbs over the 100 iteration the NRMSE increases 

#It is worthy to note that when the regularization parameter is so small to converge to eigenvalues takes time

#we will use the modal empx which has the function nad produces number of iterations and the normalized eigenvector


    betta,iteration=eig(THICK_RID) #This the 
    betta=betta/LA.norm(betta)


#in case you wanted to plot the eigenvectors 
    #betta_array=np.array(betta)[0]
    #la=range(len(betta_array))
    #plt.plot(la,betta_array)
    #plt.xlabel(' # of wavelength')
    #plt.ylabel('coffecient of the basis vector betta')
    #plt.show()



#********************************************
#***Estimating the functional relationship***
#********************************************

#creating the projected values of the slices where  x_proj will be a vector composed of the projected data 
#Not that xh_bar is a matrix which contains the average of each slice so by simply doing matrix multiplication the newly transformed vector will be the projected data that we will use to do interpolation later on.
    x_proj=xh_bar*betta.T

    x_proj=np.array(x_proj.T)[0]
    

    #plt.plot(x_proj,PARM,'*')
    #plt.xlabel("x-proj")
    #plt.ylabel("PARM")
    #plt.show()
    


#Now we will interpolate some observed data we can use our synthetic specra as an initial test
#Or late we can add some noise to it.
#I will select some random values from the grid load them and project them to have
#a set of observed spectra for testing



#************************************************************************
#******generating the test data to choose the regularization vector *****
#************************************************************************

    x_obs=[]
    y_theo=[]  
	#using the select list which contains the indexes of some slected spectra 
	#will help us to choose the best regularization parameter and we will exytract and project the observed data 
	#in the work of wael there was a graph for noise so to choose the best regularizatio parameter we should add that noise and do tests	 	based on that
	#This will be fed by the list with tuples called "lister which contains the test data    

    for i in range(len(select)):
        a=np.matrix(select[i][6])
        a=float(a*betta.T)
        x_obs.append(a)
        
        b=np.matrix(select[i][parm_i])
        y_theo.append(float(b))


    

    y_obs=sor_interp(x_obs,x_proj,PARM)
    y_theo=np.array(y_theo)

    



#Now we should test or results based on the criteria of NRMSE and SIRC 
    
    y_theo_mean=np.mean(y_theo)
    A=(y_obs - y_theo)**2
    B=(y_theo - y_theo_mean)**2

    A=np.sum(A)
    B=np.sum(B)

    
    NRMSE=np.sqrt(A/B)  

    return NRMSE,betta,y_obs,iteration,y_theo



######################################################################################################################

#radial velocity correction function where it return the flux of the corrected spectrum

def rv_correct(temp_lam,temp_spec,obs_lam,obs_spec):
# Plot template and data
#plt.title("Template (blue) and data (red)")
#plt.plot(temp_lam, temp_spec, 'b.-')
#plt.plot(obs_lam, obs_spec, 'r.-')
#plt.show()


###############################################################
####            Carry out the cross-correlation.           ####
####The RV-range is -30 - +30 km/s in steps of 0.6 km/s.   ####
#### The first and last 20 points of the data are skipped. ####
###############################################################
    rv, cc = pyasl.crosscorrRV(obs_lam, obs_spec, temp_lam, temp_spec, -30., 30., 30./50., skipedge=20)

# Find the index of maximum cross-correlation function
    maxind = np.argmax(cc)

    print "Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s"
    if rv[maxind] > 0.0 : print " A red-shift with respect to the template"
    if rv[maxind]<0.0 : print " A blue-shift with respect to the template"
    if rv[maxind]<1e-10 : print "No shift"

    #plt.plot(rv, cc, 'bp-')

    #plt.plot(rv[maxind], cc[maxind], 'ro')
    #plt.show()


    c=3.0*10**5 #speed of light
    RV_wavelength=obs_lam*(1.-rv[maxind]/c)

    try:
        f=interpolate.interp1d(RV_wavelength,obs_spec,kind='linear')
        return f(obs_lam)
    except ValueError:
        f=interpolate.interp1d(RV_wavelength,obs_spec,kind='linear',bounds_error=False,fill_value=1)
        return f(obs_lam)

####################################################################################################################
 
def DER_SNR(flux):
   

   """
   DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
               definition set forth by the Spectral Container Working Group of ST-ECF,
	       MAST and CADC. 

               signal = median(flux)      
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
	       snr    = signal / noise
               values with padded zeros are skipped

   USAGE       snr = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy      
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum 
	       as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
	         more pixels can be approximated by a straight line
 
               For most spectra, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               * Software:
	       www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
   """
   from numpy import array, where, median, abs 

   flux = array(flux)

   # Values that are exactly zero (padded) are skipped
   flux = array(flux[where(flux != 0.0)])
   n    = len(flux)      

   # For spectra shorter than this, no value can be returned
   if (n>4):
      signal = median(flux)

      noise  = 0.6052697 * median(abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

      return float(signal / noise)  

   else:

      return 0.0

###################################################################################################################

def noise(f,sn):
    """Function to create Gaussian noise. f should be a numpy array, sn is the desired Signal to Noise ratio"""
    if isinstance(f,np.ndarray) is False:
	raise RuntimeError('Array should be a numpy array')
    z=f.copy()
    for i in np.arange(np.size(z)):
	r=100
	while (r>1.0 or r<0.0):
	    x=2*rd.uniform(0,1)-1
	    y=2*rd.uniform(0,1)-1
	    r=x**2+y**2
	gaussnoise=x*np.sqrt(-2*np.log(r)/r)
	z[i]=z[i]+(1./sn)*gaussnoise
	if z[i]<0.0:
	    z[i]=0.0
    return z



###################################################################################################################

########################################################
#getting the data of the synthetic spectra to use later#
########################################################

#Here we have the list box which contains the values lists of the iterated paremters sorted in increase=

   

#And since we are going to find the average of each slice which will be equivelent to the number of slices     


#We will create the SIR matrix Sigma*gamma but first we need the gamma since ssigma is ready

####################################
######Sorting the sigma matrix######
####################################

#at the end we will iterate on the 4 parameters but for now our list will only be for vsini

   

#for k in range(4):





def calc(grid_n,name_grid,lister):
    lam=len(grid_n[0][0]) #lam =12 PCA coefficient numbers
    sz=len(grid_n) #number of syn spec used
    crate={}
        
    for k in range(5):    
        box=[]
        grid_n=sorted(grid_n,key=lambda tup: tup[k+1])
        for i in range(len(grid_n)):
            box.append(grid_n[i][k+1])

       
        
            
        sorted_grid=np.zeros((sz,lam))

        for i in range(sz):
            sorted_grid[i]=grid_n[i][0]
    
    
    
    
    

    ###################################
    ###Creating the empirical matrix###
    ###################################   
    
    #using the new reduced matrix we will find a new covariance matrix
        sorted_grid=np.matrix(sorted_grid)
        x_bar,sig=cov_mat(sorted_grid)    
    

    
    ###########################
    ###Creating within slice###
    ###########################   
    
    
        X_h_bar=trenche_mean(sorted_grid,box)
        gam=with_cov_mat(X_h_bar,x_bar) 
    
        del sorted_grid

        box=list(set(box))
        box=sorted(box)
    
########################################################################    
    
####################################################### 
##Minimizing delta using the goldden section method ###    
#######################################################

#Now we will work to find the projection direction of the Thikhonov-ridge matrix 
#We will do it by the golden section rule on alogarithmic scale 
   

    #step=0.0001
    #start=0
    #end=0.003
    ##this is the regularization parameter delta loop which we will later on interplay
    #loop=np.arange(start,end+step,step)

        a=-30
        b=5
        c=(-1+np.sqrt(5))/2 
        delta1=c*a+(1-c)*b
        delta2=(1-c)*a +c*b
    
        def f(delta):
            return GRSIR(10**(delta),X_h_bar[0],box,lister,sig,gam,grid_n,k+1)
        
        try:
            f1=f(delta1)
            f2=f(delta2)    
        
        except FloatingPointError:
            crate[name_grid[k]]=['N/A','N/A','N/A','N/A','N/A','N/A','N/A']
            continue
    #this is the regularization parameter delta loop which we will later on interplay
    
        for j in range(1000):
            if f1<f2:
                b=delta2
                delta2=delta1
                f2=f1
                delta1=c*a +(1-c)*b
                f1=f(delta1)        
        
            else:
                a=delta1
                delta1=delta2
                f1=f2
                delta2=(1-c)*a +c*b
                f2=f(delta2)
            
            
                if abs(a-b)<1e-5 :
                    break

            
    
        NRMSE,bet,y_obs,xyz,y_theor=GRSIR_best(10**a,X_h_bar[0],box,lister,sig,gam,grid_n,k+1)
        

        crate[name_grid[k]]=[y_theor,y_obs,y_theor-y_obs,bet,box,X_h_bar[0],NRMSE]          





            
    return crate   
	






