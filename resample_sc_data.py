import numpy as np
from scipy.io import loadmat, savemat

scfile_orig='example_data_SCstreamlinecount.mat'
scfile_resamp='example_data_SCresampled.mat'

#read in data saved as cell array of ROIxROI square matrices of raw streamline counts
SCorig=loadmat(scfile_orig,simplify_cells=True)['sc']

#extract lower triangular, excluding diagonal (equivalent to triu(k=1) in matlab)
trimask=np.tril_indices(SCorig[0].shape[0],k=-1)
SCtri=np.vstack([x[trimask] for x in SCorig])

resamp_mean=0.5
resamp_stdev=0.1

SCresamp=np.zeros(SCtri.shape)

for i in range(SCtri.shape[0]):
    sc=SCtri[i,:]
    scmask=sc>0
    u,uidx=np.unique(sc[scmask],return_inverse=True) #sorted ascending
    r=np.random.randn(len(u))*resamp_stdev + resamp_mean #gaussian random vector
    r=np.sort(r) #sorted ascending
    
    scnew=np.zeros(sc.shape)
    scnew[scmask]=r[uidx]
    SCresamp[i,:]=scnew

savemat(scfile_resamp,{'sc':SCresamp},format='5',do_compression=True)
