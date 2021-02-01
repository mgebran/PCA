import os
import PCA_SIR_Vmicr_2018 as PCA
import numpy as np
import pylab as plb
from pylab import *
import time
import pyfits
import starlink.Atl as Atl
from scipy import interpolate
import random
import der_snr
from PyAstronomy import pyasl
from astropy.io import fits
#import empx
import time

dump='Dumpfile-lambda4450.00-4989.95-teff5000.0-6200.0-100-logg4.00-5.00-0.10-vsini0-14-meta-0.40-0.40-0.20-Vmicr1.00-1.00-1.00-Resolution31500'
Resolution=31500
#result_path="results/"

result_file="Gstars-results.dat"
temp=np.loadtxt("Template/atSS02970-t005500g4.20-vrot000res031500-met+0.0-micr1.0s48.dat")

lst=np.loadtxt("filelist",dtype=str)


path_dump=PCA._path+dump
PCA.setup(dump,PCA.lambda_polar,Resolution,"synspec")




lst_size=len(lst)
ii=0

for i in lst:
	print str(ii+1)+"/"+str(np.size(lst))
	star=lst[ii][12:]
	spec=np.loadtxt(i)

	noise=der_snr.DER_SNR(spec[:,1])
	radialv, cc = pyasl.crosscorrRV(spec[:,0], spec[:,1], temp[:,0], temp[:,1], -200., 200., 0.1, skipedge=200)
	maxind = np.argmax(cc)
	print "Cross-correlation function is maximized at dRV = ", radialv[maxind], " km/s for", str(star), "with a noise of", round(noise,4)

	spec_rvcor=PCA.rv_cor(spec[:,1],spec[:,0],float(radialv[maxind]))
	index_c,specc=PCA.inversion(spec_rvcor,cont=True)
	

	# In case we want to save the spectra with the corrected normalization and the best fit
	np.savetxt("corrected_gaz/"+str(star)+"_polar",np.array([PCA.wavelength,specc,PCA.getspec(index_c)]).T) 

	sets=PCA.dif #get the Chi_squares 
        sets_sort=np.argsort(sets)#This commands sorts the index of the spectra based on the nearest fit so the 1st elemnt is the index of the synthetic spectra with the nearest neighbor
	
	#In case we want to save the best indexes
	#np.savetxt(result_path+"corrected_gaz_best_indexes/"+str(star)+"_indexes",sets_sort,fmt="%.0f")
	
	teff=PCA.getparam(int(index_c))[0]
	Logg=PCA.getparam(int(index_c))[1]
	vsini=PCA.getparam(int(index_c))[2]
	metallicity=PCA.getparam(int(index_c))[3]
	Vmicr=PCA.getparam(int(index_c))[4]

	print "The values inverted from the PCA are:\n"
	print "Teff="+"\t"+str(teff)+"\n"
	print "logg="+"\t"+str(Logg)+"\n"
	print "vsini="+"\t"+str(vsini)+"\n"	
	print "[Fe/H]="+"\t"+str(metallicity)+"\n"	
	print "Micro="+"\t"+str(Vmicr)+"\n"
	

	flux=path_dump+'/coeff_'+dump
	Teffec=path_dump+'/teff_'+dump
	vrot=path_dump+'/vsini_'+dump
	log_g=path_dump+'/logg_'+dump
	metal=path_dump+'/meta_'+dump
	micro_t=path_dump+'/micro_'+dump
	PCA_eig=np.loadtxt(path_dump+'/eigen_'+dump)
	PCA_mean=np.loadtxt(path_dump+'/mean_'+dump)	
	grid=PCA.grider(flux,Teffec,log_g,vrot,metal,micro_t)
	name_parms=['Teff','log(g)','vsini','[Fe/H]', 'micro']
	parm_unit=['Kelvin' , 'dex' , 'Km/s' ,'dex', 'Km/s']
	

	A_T=10000.
	A_L=10000.
	A_V=10000.
	A_M=10000.
	A_MI=10000.
	iters=np.arange(20,100,10)
	for sets in iters:
   		print "RUN--> ",str(sets)," synthetic spectra" 
    		print "####################################"

    		file_output={}
    		select=15
    
    		obs=specc
    


       		i=0
        	index=[]        
        	for line in sets_sort:
            		index.append(int(line))
            		i=i+1
            		if i==sets:break
       			    
    
        	pca_obs=specc-PCA_mean
        	pca_obs=np.dot(pca_obs,PCA_eig)    
    
    
    
        	grid_new=[]  
        	for j in index:
            		grid_new.append(grid[int(j)])
        

################################################################    
#creating the reduced training data set using PCA best indexes##
################################################################

    
#Since this data set is reduced we need to recontruct it to add some noise on it 
#Based on the signal to noise of each observed psectra we can add a random noise  
#I am using a reduced data set to win time so I need to reconstruct the data   
#The new grid tha will be used are the 50 1st nearest fits and we will use 25  do GRSIR    
        	SNR=PCA.DER_SNR(specc)
        	noise_box=[]
        	test_index=random.sample(xrange(1,sets),select)
        	for i in test_index:

            		spect=np.dot(grid_new[i][0],PCA_eig.T)+PCA_mean #Expand to initial dimension
            
            		val=PCA.noise(spect,SNR)# Add the noisy
            
            		spectra_n=val-PCA_mean #remove the mean
            		spectra_n=np.dot(spectra_n,PCA_eig)  #reproject to get a reduced noisy version          
            		bag=grid_new[i][0],grid_new[i][1],grid_new[i][2],grid_new[i][3],grid_new[i][4],grid_new[i][5],spectra_n
            		noise_box.append(bag)


        	del bag,val

        
        	doup=PCA.calc(grid_new,name_parms,noise_box)
        	print str(ii)+'/'+str(lst_size)+' ---> '+ str(sets) +' ---> '+ 'iterations'
        	print '*********************************************************'

              
 ##########################
 #Estimation of parameters# 
 ##########################
  
          	results=[]
        	for z in name_parms:
                    
            		if doup[z][0]=='N/A':
                		val='N/A','N/A'
                		results.append(val)
               			continue
            
            		x_obs=float(doup[z][3]*np.matrix(pca_obs).T)
            		x_pr=doup[z][5]*doup[z][3].T
            		x_pr=np.array(x_pr.T)[0]
            		val=PCA.sor_interp(x_obs,x_pr,doup[z][4]) , doup[z][6] 
        #here I will save the name of files as dicts and each key will have a list containing tuples  of the stellar fundemantal parameter and its corresponding NRMSE
            		results.append(val)
        
        
        	#In case we want to save the files of eaxh iteration
        	#o=open(result_path+str(sets)+".dat","a")
        	for z in range(5):
            		print name_parms[z]+' = ' , results[z][0] , parm_unit[z] , '        NRMSE = ' , results[z][1]
            		if results[z][0]=='N/A': print "Raylaigh quotient encountered a 0 by 0 division for "+name_parms[z]
        	file_output[ii]=results 
		if (A_T > results[0][1]):
			teff=results[0][0]
			A_T=results[0][1]
 		

		if (A_L > results[1][1]):
			Logg=results[1][0]
			A_L=results[1][1]

		if (A_V > results[2][1]):
			vsini=results[2][0]
			A_V=results[2][1]

		if (A_M > results[3][1]):
			metallicity=results[3][0]
			A_M=results[3][1]

		if (A_MI > results[4][1]):
			Vmicr=results[4][0]
			A_MI=results[4][1]

        
        	
	print	"For the star"+"\t"+ str(ii)+"\t"+ str(star)+"\t"+"Teff="+"\t"+ str(round(float(teff),3))+"\t"+"logg="+"\t"+ str(round(float(Logg),3))+"\t"+"vsini="+"\t"+ str(round(float(vsini),3))+"\t"+"meta="+"\t"+ str(round(float(metallicity),3))+"\t"+"Vmicr="+"\t"+ str(round(float(Vmicr),3))+"\n"
	o=open(result_file,"a")
	o.write(str(star)+"\t"+str(PCA.getparam(int(index_c))[0])+"\t"+str(round(float(teff),3))+"\t"+str(PCA.getparam(int(index_c))[1])+"\t"+str(round(float(Logg),3))+"\t"+str(PCA.getparam(int(index_c))[2])+"\t"+str(round(float(vsini),3))+"\t"+str(PCA.getparam(int(index_c))[3])+"\t"+str(round(float(metallicity),3))+"\t"+str(PCA.getparam(int(index_c))[4])+"\t"+ str(round(float(Vmicr),3))+"\n")
	o.close()


	ii+=1

#os.system("./mail.sh"))
