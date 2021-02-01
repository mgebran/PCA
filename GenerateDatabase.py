###############################################################################################
#                 THIS SCRIPT TAKES DUMPFILE AS INPUT AND PERFORMS PCA                        #
# IT CREATES FILES IN specified DIRECTORY OF COEFFICIENTS, EIGENVECTORS, AND OTHER PARAMETERS #
#                            EXECUTE ONLY ONCE FOR EACH DATABASE                              #
###############################################################################################

import matplotlib as plb
import numpy as np
import pickle
import time

t=time.time()

######################
#  READING THE DATA  #
######################

Name_dp="DumpfileWEAVE-6200-6750-7000-15000-500-4-5-0.5-vsini-0-300-1-1-0.50-Vmicr2-Resolution20000" # Enter name of dumpfile
path="/home/mgebran/Desktop/" # Enter path to dumpfile


Fic=open(path+Name_dp,'r')
grid=pickle.load(Fic)
Fic.close()
print "\n\n\n\n\n"
print "==========================================================================="
print "Synthetic data read from pickle"
print "===========================================================================\n"

nmod=len(grid) # Number of spectra in learning database
nwav=len(grid[0][0]) # Numer of data point in each spectra

#--- common wavelength grid

mat=np.zeros((len(grid), nwav), 'Float32') # nmod by nmwav matrix of learning database matrix
teff=[] # Teff of each spectrum
logg=[] # logg of each spectrum
vrot=[] # vrot of each spectrum
meta=[] # vrot of each spectrum
micro=[]

for i in np.arange(nmod):
    mat[i,:]=grid[i][0]
    teff.append(grid[i][1])
    logg.append(grid[i][2])
    vrot.append(grid[i][3])
    meta.append(grid[i][4])
    micro.append(grid[i][5])
print "==========================================================================="
print  "Matrix of data, vectors of Teff and logg done, proceeding to SVD"
print "===========================================================================\n"

'''
plb.pyplot.imshow(mat, aspect='auto')
plb.pyplot.colorbar()
print "==========================================================================="
print "X-axis represent wavelength"
print "===========================================================================\n"
'''
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
print "Final step: saving files:"
print "===========================================================================\n"

np.savetxt(path+"coeff_"+Name_dp,p)
np.savetxt(path+"eigen_"+Name_dp,e[:,:12])
np.savetxt(path+"teff_"+Name_dp,teff)
np.savetxt(path+"logg_"+Name_dp,logg)
np.savetxt(path+"mean_"+Name_dp,mn)
np.savetxt(path+"vsini_"+Name_dp,vrot)
#np.savetxt(path+"mat_"+Name_dp,mat)
np.savetxt(path+"meta_"+Name_dp,meta)
t=(time.time()-t)/60
print "Calculation took: (mins)"
print t
