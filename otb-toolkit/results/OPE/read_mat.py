import scipy.io as sio
import numpy as np
matfn="./Trellis_KCF.mat"
data=sio.loadmat(matfn)
print(data["results"])
